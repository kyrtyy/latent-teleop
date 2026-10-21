"""PushT at 30 ms ticks: full-state access, render-from-state, contact test, and a scripted operator."""
import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
from gym_pusht.envs.pusht import PushTEnv

TICK_MS = 30
IMG = 224
GOAL = np.array([256.0, 256.0, np.pi / 4])  # PushTEnv._setup goal_pose (block origin x, y, angle)
R_AGENT = 15
KP, KV, SUB_DT, SUBSTEPS = 100.0, 20.0, 0.01, 3  # PushTEnv agent PD gains and substeps per tick

# state vector layout (10): agent pos, agent vel, block pos, block angle, block vel, block angular vel
AGENT, AGENT_V, BLOCK, ANGLE = slice(0, 2), slice(2, 4), slice(4, 6), 6


def make_env(seed=0):
    env = PushTEnv(obs_type="state", observation_width=IMG, observation_height=IMG)
    env.control_hz = 33.0  # step() runs int(1 / (0.01 * 33)) = 3 substeps of 10 ms = one 30 ms tick
    env.reset(seed=seed)
    return env


def full_state(env):
    a, b = env.agent, env.block
    return np.array([*a.position, *a.velocity, *b.position, b.angle, *b.velocity, b.angular_velocity])


def in_contact(env):
    # ponytail: info["n_contacts"] is unusable (wrong divisor, counts walls), so test agent vs block directly
    agent = next(iter(env.agent.shapes))
    return any(agent.shapes_collide(s).points for s in env._block_shapes)


def render(env, s):
    """The frame depends only on agent position and block pose (goal is fixed), so frames are never stored."""
    env.block.angle = float(s[ANGLE])  # angle first: pymunk rotates about the CoG, which would move the origin
    env.agent.position, env.block.position = s[AGENT].tolist(), s[BLOCK].tolist()
    env.space.reindex_shapes_for_body(env.agent)
    env.space.reindex_shapes_for_body(env.block)
    return env._render()  # visualize=False: no action marker drawn


def agent_pd(pos, vel, targets):
    """Integrate the agent's PD controller exactly as PushTEnv.step does. The agent is KINEMATIC, so this is exact.
    pos, vel (..., 2); targets (..., K, 2) -> positions after each tick (..., K, 2)."""
    pos, vel, out = pos.astype(float).copy(), vel.astype(float).copy(), []
    for a in np.moveaxis(targets, -2, 0):
        for _ in range(SUBSTEPS):
            vel += (KP * (a - pos) - KV * vel) * SUB_DT
            pos += vel * SUB_DT
        out.append(pos.copy())
    return np.stack(out, -2)


# --- scripted operator -------------------------------------------------------------------------------

T_OUTLINE = sg.Polygon([(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)])  # add_tee, scale 30
T_COM = np.array([0.0, 45.0])  # mean of the two rectangles' centroids, as add_tee sets center_of_gravity


def _push_candidates(spacing=8):
    """Points on the T outline (block frame) with outward normals, where the agent can stand clear of the block."""
    pts, nrm = [], []
    v = np.array(T_OUTLINE.exterior.coords)
    for p0, p1 in zip(v[:-1], v[1:]):
        length = np.linalg.norm(p1 - p0)
        d = (p1 - p0) / length
        n = np.array([d[1], -d[0]])
        if T_OUTLINE.contains(sg.Point((p0 + p1) / 2 + n)):
            n = -n
        for s in np.arange(spacing / 2, length, spacing):
            p = p0 + d * s
            if T_OUTLINE.distance(sg.Point(p + n * (R_AGENT + 6))) >= R_AGENT + 2:
                pts.append(p)
                nrm.append(n)
    return np.array(pts), np.array(nrm)


PTS, NRM = _push_candidates()


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class ScriptedOperator:
    """Pushes the T toward the goal using only the scene shown on screen (`s` = the displayed, delayed frame).
    The cursor is local and moves at once; the robot follows it through the delayed uplink. Never reads the env."""

    def __init__(self, rng, start):
        self.rng = rng
        self.params = dict(
            speed=rng.uniform(8, 16),  # cursor px per tick
            noise=rng.uniform(0, 1.5),  # cursor jitter px
            push_ticks=int(rng.integers(6, 16)),  # how long one push is committed before re-looking
            push_depth=rng.uniform(20, 40),  # how far past the contact point the cursor aims
            explore=rng.uniform(0, 0.15),  # chance a re-plan picks a random push (data diversity)
            hesitate=rng.uniform(0, 0.01),  # chance per tick to freeze for 5-20 ticks
        )
        self.cursor = np.asarray(start, float).copy()
        self.mode, self.until, self.pause_until = "approach", 0, 0
        self.u, self.target = None, None

    def _plan(self, s):
        pos, th = s[BLOCK], s[ANGLE]
        R = _rot(th)
        p, n = PTS @ R.T + pos, NRM @ R.T
        u, com = -n, R @ T_COM + pos
        dp = GOAL[:2] - pos
        dist = np.linalg.norm(dp)
        r = p - com
        torque = (r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]) / 60
        score = u @ dp / max(dist, 1e-6) * min(dist / 40, 1) + torque * np.clip(wrap(GOAL[2] - th) / 0.5, -1, 1)
        a = p + n * (R_AGENT + 6)
        ok = np.all((a > 20) & (a < 492), axis=1)
        score[~ok] = -np.inf
        if ok.any() and self.rng.random() < self.params["explore"]:
            i = self.rng.choice(np.flatnonzero(ok))
        else:
            i = int(np.argmax(score))
        return p[i], u[i], a[i], com

    def _push(self, p, u, t):
        self.mode, self.until, self.u = "push", t + self.params["push_ticks"], u
        self.target = p + u * self.params["push_depth"]

    def _route(self, s, agent, a, com):
        c, si = np.cos(s[ANGLE]), np.sin(s[ANGLE])
        t_world = sa.affine_transform(T_OUTLINE, [c, -si, si, c, *s[BLOCK]])
        if sg.LineString([agent, a]).distance(t_world) >= R_AGENT + 1:
            return a
        off = agent - com
        d = np.linalg.norm(off)
        if d < 125:  # too close to go around: back away radially first
            return com + off / max(d, 1e-6) * 135
        ang = np.arctan2(off[1], off[0])
        ang += np.clip(wrap(np.arctan2(a[1] - com[1], a[0] - com[0]) - ang), -0.5, 0.5)
        return com + 135 * np.array([np.cos(ang), np.sin(ang)])

    def act(self, s, t):
        if t < self.pause_until:
            return self.cursor.copy()
        if self.rng.random() < self.params["hesitate"]:
            self.pause_until = t + int(self.rng.integers(5, 20))
            return self.cursor.copy()
        agent = s[AGENT]
        if self.mode == "push" and t >= self.until:
            p, u, a, com = self._plan(s)
            if u @ self.u > 0.9 and np.linalg.norm(p - agent) < 2 * R_AGENT:
                self._push(p, u, t)  # same push still makes sense: keep going without backing off
            else:
                self.mode = "approach"
        if self.mode == "approach":
            p, u, a, com = self._plan(s)
            if np.linalg.norm(agent - a) < 8:  # judged on the *perceived* agent position
                self._push(p, u, t)
            else:
                desired = self._route(s, agent, a, com)
        if self.mode == "push":
            desired = self.target
        step = desired - self.cursor
        step *= min(1.0, self.params["speed"] / max(np.linalg.norm(step), 1e-6))
        self.cursor = np.clip(self.cursor + step + self.rng.normal(0, self.params["noise"], 2), 0, 512)
        return self.cursor.copy()


if __name__ == "__main__":
    env = make_env(seed=1)
    # contact test fires when the agent overlaps the block, and not when far away
    s = full_state(env)
    env.agent.position = (s[BLOCK] + _rot(s[ANGLE]) @ np.array([0.0, 15.0])).tolist()  # inside the bar
    env.space.reindex_shapes_for_body(env.agent)
    assert in_contact(env), "contact test missed an overlap"
    env.reset(seed=1)
    env.agent.position = (10.0, 10.0) if np.linalg.norm(s[BLOCK] - 10) > 150 else (500.0, 500.0)
    env.space.reindex_shapes_for_body(env.agent)
    assert not in_contact(env)

    # block is quasi-static (space damping 0): once the agent backs off, the block does not drift
    env.reset(seed=2)
    s = full_state(env)
    com = _rot(s[ANGLE]) @ T_COM + s[BLOCK]
    start = com + np.array([0.0, -140.0])
    env.agent.position = start.tolist()
    moved = False
    for i in range(60):
        env.step(start + (com - start) * min(1, i / 30))
        moved |= np.linalg.norm(full_state(env)[BLOCK] - s[BLOCK]) > 1
    assert moved, "push test never moved the block"
    for _ in range(10):
        env.step(start)
    b0 = full_state(env)[BLOCK]
    for _ in range(30):
        env.step(start)
    assert np.linalg.norm(full_state(env)[BLOCK] - b0) < 0.1, "block drifts without contact"

    # render-from-state reproduces the live render exactly
    env.reset(seed=3)
    for _ in range(20):
        env.step(np.array([300.0, 200.0]))
    live, st = env._render(), full_state(env)
    env.reset(seed=4)
    assert np.array_equal(render(env, st), live), "render(state) differs from live render"

    # PD replay matches the simulator's agent exactly (this is what the kinematic baseline relies on)
    env.reset(seed=5)
    s0, targets = full_state(env), np.random.default_rng(0).uniform(50, 450, (40, 2))
    real = []
    for a in targets:
        env.step(a)
        real.append(full_state(env)[AGENT])
    assert np.allclose(agent_pd(s0[AGENT], s0[AGENT_V], targets), real, atol=1e-6), "agent PD replay drifted"
    print(f"sim.py checks passed ({len(PTS)} push candidates)")

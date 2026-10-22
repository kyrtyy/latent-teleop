"""C2 ghost: what the operator perceives under delay, built only from what the operator can observe:
the frames received so far (each stamped with the robot's capture tick and the command it executed the tick
before), and the commands the operator itself has sent.

From the newest frame (tick tau) the ghost predicts the scene at T = now + estimated uplink delay: the tick at which
the command being sent now will start to act. Commands still in flight are placed on the robot's timeline with the
same estimate, and a command overtaken by a newer one is dropped, as the robot does.

kinds:
  none       the delayed frame (what the operator sees today)
  kinematic  agent replayed exactly through its PD controller on the timeline, block held (a Smith predictor)
  learned    kinematic agent + block from the latent world model, used only when the agent's predicted path comes
             within reach of the block; otherwise the block is held (hold is exact in free space)
  oracle     true physics from the full state at tau on the same estimated timeline (upper bound for the model)
"""
import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
import torch

from sim import AGENT, AGENT_V, ANGLE, BLOCK, R_AGENT, T_OUTLINE, agent_pd, full_state, make_env, set_state
from world_model import decode

H = 4  # context frames the world model takes


class Ghost:
    def __init__(self, kind, max_k=60, window=30):
        self.kind, self.max_k, self.window = kind, max_k, window
        self.lags, self.done = [], set()  # observed ages of executed commands (ticks); frame ticks already read
        self.stats = {"capped": 0, "fallback": 0, "learned": 0, "ticks": 0}
        self.env = make_env() if kind == "oracle" else None

    def observe(self, received):
        """Frame c says the command sent at tick e ran during tick c-1: that command's age when executed is c-1-e.
        This, not the per-packet delay, is what the timeline needs: under jitter the robot runs the newest command
        that has landed, and slow packets get overtaken, so executed commands are younger than the median packet."""
        for c in sorted(received.keys() - self.done):
            if received[c][1] >= 0:
                self.lags.append(c - 1 - received[c][1])
            self.done.add(c)

    def lag(self, t, tau):
        if not self.lags:
            return t - tau  # nothing measured yet: assume the uplink is as slow as the downlink
        return int(np.median(self.lags[-self.window:]))

    def timeline(self, t, tau, frame, sent, lag):
        """Commands the robot will execute during ticks tau .. T-1, T = t + lag."""
        e, cmd = frame[1], frame[2]  # executing during tau-1
        out, s = [], max(e, -1)
        for j in range(tau, t + lag):
            while s + 1 < len(sent) and s + 1 + lag <= j:  # newest command that has landed by tick j
                s += 1
            if s > e:
                e, cmd = s, sent[s]
            out.append(cmd)
        return np.array(out).reshape(-1, 2)

    def __call__(self, t, received, tau, shown, sent):
        """-> (perceived state, or a request dict for the batched driver; the tick that state is for)."""
        self.stats["ticks"] += 1
        if self.kind == "none" or tau < 0:
            return shown, tau
        self.observe(received)
        lag = self.lag(t, tau)
        if t + lag - tau > self.max_k:
            lag, self.stats["capped"] = self.max_k - t + tau, self.stats["capped"] + 1
        T = t + lag
        if T <= tau:
            return shown, tau
        u = self.timeline(t, tau, received[tau], sent, lag)
        if self.kind == "oracle":
            set_state(self.env, shown)
            for c in u:
                self.env.step(c)
            return full_state(self.env), T
        path = agent_pd(shown[AGENT], shown[AGENT_V], u)
        seen = shown.copy()
        seen[AGENT], seen[AGENT_V] = path[-1], 0.0
        if self.kind == "kinematic" or not self._reaches_block(shown, path):
            return seen, T
        # newest complete H-frame window: under jitter, frames just before tau may still be in flight
        end = next((e for e in range(tau, tau - 10, -1) if all(c in received for c in range(e - H + 1, e + 1))), None)
        if end is None or T - end > self.max_k:
            self.stats["fallback"] += 1  # hold the block this tick
            return seen, T
        ctx = list(range(end - H + 1, end + 1))
        u = self.timeline(t, end, received[end], sent, lag)  # commands from the window's last frame on
        self.stats["learned"] += 1
        acts = [received[c + 1][2] for c in ctx[:-1]]  # frame c+1 stamps the command executed during tick c

        def finish(p):  # p = probe output for the predicted frame at T: agent xy, block xy, cos, sin, contact
            _, b, th, _ = decode(torch.as_tensor(p)[None])
            seen[BLOCK], seen[ANGLE] = b[0].numpy(), float(th[0])
            return seen

        return {"ctx": ctx, "states": [received[c][0] for c in ctx], "actions": np.concatenate([acts, u]),
                "k": len(u), "finish": finish}, T

    @staticmethod
    def _reaches_block(s, path, margin=10):
        c, si = np.cos(s[ANGLE]), np.sin(s[ANGLE])
        t_world = sa.affine_transform(T_OUTLINE, [c, -si, si, c, *s[BLOCK]])
        return sg.LineString(np.vstack([s[AGENT], path])).distance(t_world) < R_AGENT + margin


if __name__ == "__main__":
    from sim import ScriptedOperator
    from teleop import Delay, Link, run_episode

    def ep(seed, kind, up, down, ticks=300):
        env, rng = make_env(seed), np.random.default_rng(seed)
        g = Ghost(kind)
        return run_episode(env, ScriptedOperator(rng, full_state(env)[AGENT]), Link(up), Link(down), ticks, g), g

    # zero delay: every ghost is the live frame
    for kind in ["kinematic", "oracle"]:
        L, _ = ep(0, kind, lambda: 0.0, lambda: 0.0)
        assert np.allclose(L["perceived"], L["state"][:-1]), kind

    # constant 150 ms each way: the lag estimate is exact, and the timeline and kinematic agent match what happened
    L, g = ep(1, "kinematic", lambda: 150.0, lambda: 150.0)
    assert g.lags and set(g.lags) == {5}, g.lags[:10]
    T, ok = L["ghost_tick"], L["ghost_tick"] < len(L["state"])
    assert np.all(T[10:] == np.arange(len(T))[10:] + 5)  # T = now + uplink lag
    err = np.linalg.norm(L["perceived"][ok, AGENT] - L["state"][T[ok], AGENT], axis=-1)
    assert err[20:].max() < 1e-6, err.max()  # past the start-up ticks, the kinematic agent is exact

    # oracle: true physics on the right timeline reproduces the future frame (pymunk solver warm start aside)
    L, _ = ep(2, "oracle", lambda: 150.0, lambda: 150.0)
    T, ok = L["ghost_tick"], L["ghost_tick"] < len(L["state"])
    err = np.linalg.norm(L["perceived"][ok, BLOCK] - L["state"][T[ok], BLOCK], axis=-1)
    assert np.median(err[20:]) < 0.1 and err[20:].max() < 2, (np.median(err), err.max())

    # jittery bursty links: the estimate tracks the true age of executed commands, and the kinematic ghost places
    # the agent far better than the delayed frame does
    rng = np.random.default_rng(3)
    L, g = ep(3, "kinematic", Delay("300b", rng), Delay("300b", rng), 600)
    true_age = np.median(np.arange(len(L["exec_tick"]))[L["exec_tick"] >= 0] - L["exec_tick"][L["exec_tick"] >= 0])
    assert abs(np.median(g.lags) - true_age) <= 1, (np.median(g.lags), true_age)
    T, ok = L["ghost_tick"], (L["ghost_tick"] >= 0) & (L["ghost_tick"] < len(L["state"]))
    ghost_err = np.linalg.norm(L["perceived"][ok, AGENT] - L["state"][T[ok], AGENT], axis=-1).mean()
    shown_err = np.linalg.norm(L["state"][L["shown_tick"][ok], AGENT] - L["state"][T[ok], AGENT], axis=-1).mean()
    assert ghost_err < 0.5 * shown_err, (ghost_err, shown_err)
    print(f"ghost.py checks passed (300b: est age {np.median(g.lags):.0f} vs true {true_age:.0f} ticks; agent error "
          f"{ghost_err:.1f}px ghost vs {shown_err:.1f}px delayed frame; capped {g.stats['capped']}/{g.stats['ticks']})")

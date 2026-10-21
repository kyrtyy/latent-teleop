"""Delayed network links and the teleop loop: operator <-> robot over an uplink (commands) and a downlink (frames)."""
import heapq
import math

import numpy as np

from sim import TICK_MS, AGENT, full_state, in_contact

# one-way delay per packet: (median ms, gaussian jitter sd ms, extra ms while in a burst)
# ponytail: synthetic profiles; real WAN recordings plug in as "trace:<file>" (one delay in ms per line)
PROFILES = {
    "0": (0, 0, 0),
    "50j": (50, 15, 0),
    "150j": (150, 40, 0),
    "300j": (300, 60, 0),
    "150b": (150, 40, 250),
    "300b": (300, 60, 300),
}


class Delay:
    """Samples one packet's delay in ms. Bursts follow a two-state Gilbert-Elliott chain (~10 packets, ~9% of time)."""

    def __init__(self, profile, rng):
        self.rng, self.bad = rng, False
        if profile.startswith("trace:"):
            self.trace = np.loadtxt(profile[6:], ndmin=1)
            self.i = int(rng.integers(len(self.trace)))
        else:
            self.trace = None
            self.med, self.jit, self.burst = PROFILES[profile]

    def __call__(self):
        if self.trace is not None:
            self.i += 1
            return float(self.trace[self.i % len(self.trace)])
        self.bad = self.rng.random() < (0.9 if self.bad else 0.01)
        return max(0.0, self.med + self.jit * self.rng.standard_normal() + self.burst * self.bad)


class Link:
    """One direction of the network. Packets can arrive out of order; recv returns them sorted by send tick."""

    def __init__(self, delay):
        self.delay, self.q, self.n = delay, [], 0

    def send(self, t, payload):
        lag = math.ceil(self.delay() / TICK_MS - 1e-9)
        heapq.heappush(self.q, (t + lag, t, self.n, payload))
        self.n += 1
        return lag

    def recv(self, t):
        out = []
        while self.q and self.q[0][0] <= t:
            out.append(heapq.heappop(self.q))
        return [(sent, p) for _, sent, _, p in sorted(out, key=lambda x: x[1])]


def run_episode(env, operator, up, down, max_ticks):
    """Tick t: robot captures frame t -> operator sees the newest frame that has arrived -> operator sends a command ->
    robot executes the newest command that has arrived (holding the last one otherwise) -> physics advances 30 ms."""
    s = full_state(env)
    shown_tick, shown = -1, s.copy()  # the session opens with a snapshot of the start scene
    exec_tick, cmd_exec = -1, s[AGENT].copy()  # the robot holds its start position until a command lands
    log = {k: [] for k in ["state", "contact", "coverage", "sent", "exec", "exec_tick", "shown_tick", "up", "down"]}
    success = False
    for t in range(max_ticks):
        log["state"].append(s)
        log["contact"].append(in_contact(env))
        log["coverage"].append(env._get_coverage())
        log["down"].append(down.send(t, s))
        for cap, frame in down.recv(t):
            if cap > shown_tick:  # a frame older than the one on screen is never shown
                shown_tick, shown = cap, frame
        cmd = operator.act(shown, t)
        log["up"].append(up.send(t, cmd))
        for sent, c in up.recv(t):
            if sent > exec_tick:  # a command overtaken by a newer one is stale and dropped
                exec_tick, cmd_exec = sent, c
        _, _, success, _, _ = env.step(cmd_exec)
        s = full_state(env)
        for k, v in [("sent", cmd), ("exec", cmd_exec), ("exec_tick", exec_tick), ("shown_tick", shown_tick)]:
            log[k].append(v)
        if success:
            break
    log["state"].append(s)
    log["contact"].append(in_contact(env))
    log["coverage"].append(env._get_coverage())
    out = {k: np.array(v) for k, v in log.items()}
    out["success"] = np.array(success)
    return out


if __name__ == "__main__":
    from sim import make_env, ScriptedOperator

    def episode(seed, up_delay, down_delay, ticks=300):
        env, rng = make_env(seed), np.random.default_rng(seed)
        op = ScriptedOperator(rng, full_state(env)[AGENT])
        return run_episode(env, op, Link(up_delay), Link(down_delay), ticks)

    # out-of-order: command 0 is slow (arrives t=10), command 1 is fast (t=1); command 0 must be dropped as stale
    seq = iter([300.0, 0.0])
    link, exec_tick = Link(lambda: next(seq)), -1
    link.send(0, "old"), link.send(1, "new")
    for t in range(12):
        for sent, c in link.recv(t):
            if sent > exec_tick:
                exec_tick, cmd = sent, c
    assert cmd == "new" and exec_tick == 1

    # zero delay is an ordinary closed loop
    L = episode(0, lambda: 0.0, lambda: 0.0)
    n = len(L["exec_tick"])
    assert np.array_equal(L["shown_tick"], np.arange(n)) and np.array_equal(L["exec_tick"], np.arange(n))

    # constant 150 ms = 5 ticks each way; before anything arrives the start snapshot / start position is held
    L = episode(1, lambda: 150.0, lambda: 150.0)
    want = np.arange(len(L["shown_tick"])) - 5
    want[want < 0] = -1
    assert np.array_equal(L["shown_tick"], want) and np.array_equal(L["exec_tick"], want)

    # bursty jitter: the screen never goes back in time and never shows a frame that has not arrived (no live leak)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        L = episode(seed, Delay("300b", rng), Delay("300b", rng))
        t = np.arange(len(L["shown_tick"]))
        for key, lag in [("shown_tick", "down"), ("exec_tick", "up")]:
            k = L[key]
            assert np.all(np.diff(k) >= 0), f"{key} went backwards"
            got = k >= 0
            assert np.all(k[got] + L[lag][k[got]] <= t[got]), f"{key} used a packet before it arrived"
    print("teleop.py checks passed")

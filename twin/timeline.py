"""What every robot and the camera do at each frame of the film (numpy only, testable without Isaac Sim).

Rover: drives along the tramline from weed to weed; at each stop the UR5e pulls the weed (arm.PickPlace: down,
close, lift, over to the bin, release) and the weed drops into the bin on the deck, which fills up.
Drone: flies a lawnmower survey at constant speed with rounded turns. Jackal: drives down a maize inter-row.
Shots: a list of (name, n_frames, camera function, caption).
"""
import math

import numpy as np

import arm as armk

FPS = 30
DRIVE_T = 2.2                       # s between two weeds
BIN_SCALE = 0.6                     # pulled weeds wilt and fold up in the bin
DRONE_SPEED, DRONE_Z = 6.0, 11.0
JACKAL_X, JACKAL_Y0, JACKAL_V = 1.0, -26.0, 0.75     # drives north along the farm track beside the maize


def tz(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def rotz(a):
    return armk.rz(a)


def _yaw(p):
    """Jaw yaw that turns with the arm (weeds and the bin are round enough), so the last wrist joint stays put."""
    return math.atan2(p[1], p[0])


def _arc(p0, p1, w):
    """Point at fraction w on the path p0 -> p1: around the base (cylindrical interpolation) for long sweeps,
    straight for short vertical moves."""
    r0, r1 = math.hypot(p0[0], p0[1]), math.hypot(p1[0], p1[1])
    a0, a1 = math.atan2(p0[1], p0[0]), math.atan2(p1[1], p1[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    if abs(da) < 0.05:
        return p0 + (p1 - p0) * w
    r, a = r0 + (r1 - r0) * w, a0 + da * w
    z = p0[2] + (p1[2] - p0[2]) * w + 0.06 * math.sin(math.pi * w)        # a little hop over the deck
    return np.array([r * math.cos(a), r * math.sin(a), z])


class WeedPick:
    """One weeding cycle for the UR5e: over the weed, straight down, close, lift, over the bin, down, open, back.
    Every move follows its Cartesian path with IK solved step by step from the previous step (continuation),
    starting from one elbow-up home posture, so the arm never jumps between IK branches."""

    CLEAR = 0.22
    HOME = np.array([0.02, 0.30, 0.20])                    # tcp above the deck edge, facing the bed (arm frame)
    STEPS = (("to_src", 1.00, "home", "src_up"), ("down", 0.55, "src_up", "src"), ("close", 0.35, None, None),
             ("up", 0.55, "src", "src_up"), ("to_dst", 1.35, "src_up", "dst_up"), ("down_dst", 0.40, "dst_up", "dst"),
             ("open", 0.25, None, None), ("up_dst", 0.35, "dst", "dst_up"), ("home", 1.20, "dst_up", "home"))
    N = 20
    _home_q = None

    def __init__(self, src, dst, item):
        src, dst = np.asarray(src, float), np.asarray(dst, float)
        self.a_open, self.a_close = armk.grip_angle(item[0] + 0.035), armk.grip_angle(item[0])
        lift = np.array([0.0, 0.0, self.CLEAR])
        self.poses = {"home": self.HOME, "src_up": src + lift, "src": src, "dst_up": dst + lift, "dst": dst}
        if WeedPick._home_q is None:
            q, res = armk.ik(armk.target(self.HOME, _yaw(self.HOME)), np.radians([90, -60, 100, -130, -90, 0]),
                             iters=400)
            if res > 1e-4:
                raise ValueError(f"home pose not reached ({res:.5f})")
            WeedPick._home_q = q
        q = WeedPick._home_q
        self.paths = {}
        for name, d, k0, k1 in self.STEPS:
            if k0 is None:
                continue
            p0, p1 = self.poses[k0], self.poses[k1]
            qs = [q]
            for w in np.linspace(0, 1, self.N + 1)[1:]:
                p = _arc(p0, p1, w)
                qn, res = armk.ik(armk.target(p, _yaw(p)), q, iters=120)
                if res > 2e-3:
                    raise ValueError(f"weed pick: lost the path {k0}->{k1} ({res:.4f})")
                q = q + (qn - q + math.pi) % (2 * math.pi) - math.pi
                qs.append(q)
            self.paths[name] = np.array(qs)
        self.q_src = self.paths["down"][-1]
        self.T = sum(d for _, d, _, _ in self.STEPS)

    def at(self, t):
        """-> (joints, finger angle, weed state 'src' | 'held' | 'dst')."""
        t = min(max(t, 0.0), self.T - 1e-9)
        t0 = 0.0
        for name, d, k0, _ in self.STEPS:
            if t < t0 + d:
                u = (t - t0) / d
                break
            t0 += d
        w = armk.minjerk(u)
        order = [s[0] for s in self.STEPS]
        i = order.index(name)
        closed = i > order.index("close") or (name == "close" and u > 0.6)
        released = i > order.index("open") or (name == "open" and u > 0.3)
        state = "dst" if released else ("held" if closed else "src")
        if name == "close":
            return self.q_src, self.a_open + (self.a_close - self.a_open) * w, state
        if name == "open":
            return self.paths["down_dst"][-1], self.a_close + (self.a_open - self.a_close) * w, state
        P = self.paths[name]
        x = w * self.N
        j = min(int(x), self.N - 1)
        q = P[j] + (P[j + 1] - P[j]) * (x - j)
        a = self.a_close if (closed and not released) else self.a_open
        return q, a, state


class Rover:
    """Weeding schedule. ground(x, y) -> z; targets (n, 2) weed positions; weed_z (n,) weed base heights."""

    def __init__(self, ground, targets, weed_z, arm_mount, bin_floor, item=(0.03, 0.03, 0.04)):
        self.ground, self.mount, self.bin = ground, np.asarray(arm_mount, float), np.asarray(bin_floor, float)
        self.stops = [float(x - arm_mount[0]) for x, _ in targets]
        self.pp, self.rel = [], []
        for (x, y), wz, xs in zip(targets, weed_z, self.stops):
            base_z = ground(xs, 0.0) + arm_mount[2]
            src = np.array([0.0, y - arm_mount[1], wz - base_z + 0.03])
            dst = self.bin + [0.0, 0.0, 0.10] - self.mount
            pp = WeedPick(src, dst, item)
            self.pp.append(pp)
            # weed pose relative to the tcp at the grasp, so it hangs from the gripper exactly as it was picked
            tcp = self.world(xs) @ tz(*self.mount) @ armk.fk(pp.q_src)["tcp"]
            self.rel.append(np.linalg.inv(tcp) @ tz(x, y, wz))
        self.cycle = DRIVE_T + self.pp[0].T
        r = np.random.default_rng(5)
        self.in_bin = [tz(self.bin[0] + (-0.09 + 0.09 * (k % 3)) + r.uniform(-0.02, 0.02),
                          self.bin[1] + (-0.06 + 0.12 * ((k // 3) % 2)) + r.uniform(-0.02, 0.02),
                          self.bin[2] + 0.012 + 0.015 * (k // 6)) @ rotz(r.uniform(0, 2 * np.pi))
                       @ np.diag([BIN_SCALE, BIN_SCALE, BIN_SCALE, 1.0]) for k in range(len(targets))]

    def world(self, x):
        return tz(x, 0.0, self.ground(x, 0.0))

    def at(self, t):
        """-> rover x, arm joints, finger angle, weed index being worked on, phase time within the pick."""
        k = min(int(t // self.cycle), len(self.pp) - 1)
        tc = t - k * self.cycle
        if tc < DRIVE_T:                                     # driving to stop k (from stop k-1)
            u = armk.minjerk(tc / DRIVE_T)
            x0 = self.stops[k - 1] if k > 0 else self.stops[0] - 1.6
            x = x0 + (self.stops[k] - x0) * u
            q0 = self.pp[k - 1].at(self.pp[k - 1].T - 1e-6)[0] if k > 0 else self.pp[0].at(0.0)[0]
            q1 = self.pp[k].at(0.0)[0]
            dq = (q1 - q0 + math.pi) % (2 * math.pi) - math.pi        # same pose modulo 2 pi: take the short way
            return x, q0 + dq * u, self.pp[k].a_open, k, -1.0
        tp = min(tc - DRIVE_T, self.pp[k].T - 1e-6)
        q, a, _ = self.pp[k].at(tp)
        return self.stops[k], q, a, k, tp

    def weed_pose(self, i, t):
        """World pose of target weed i at time t, or its home pose (None means: leave it where it grew)."""
        x, q, a, k, tp = self.at(t)
        if i > k or (i == k and tp < 0):
            return None
        if i < k or tp < 0:
            done_x = x if i < k else x
            return self.world(done_x) @ self.in_bin[i]
        _, _, state = self.pp[k].at(tp)
        if state == "src":
            return None
        if state == "held":
            return self.world(x) @ tz(*self.mount) @ armk.fk(q)["tcp"] @ self.rel[i]
        # released: drop into the bin over 0.25 s
        t_rel = self._release_time(k)
        held = self.world(x) @ tz(*self.mount) @ armk.fk(self.pp[k].at(t_rel)[0])["tcp"] @ self.rel[i]
        end = self.world(x) @ self.in_bin[i]
        u = min(1.0, (tp - t_rel) / 0.25)
        P = held.copy()
        P[:3, 3] = held[:3, 3] + (end[:3, 3] - held[:3, 3]) * u * u
        P[:3, :3] *= 1.0 + (BIN_SCALE - 1.0) * u
        if u >= 1.0:
            return end
        return P

    def _release_time(self, k):
        pp = self.pp[k]
        if not hasattr(pp, "_t_rel"):
            ts = np.arange(0, pp.T, 0.01)
            pp._t_rel = next(t for t in ts if t > pp.T * 0.4 and pp.at(t)[2] == "dst")
        return pp._t_rel


class Drone:
    """Lawnmower survey at constant speed, corners rounded by Chaikin subdivision."""

    def __init__(self, xs=(-40, -32, -24, -16, -8, 0, 8, 16), y=(-24.0, 24.0), z=DRONE_Z, speed=DRONE_SPEED):
        pts = []
        for i, x in enumerate(xs):
            a, b = (y[0], y[1]) if i % 2 == 0 else (y[1], y[0])
            pts += [(x, a), (x, b)]
        P = np.array(pts, float)
        for _ in range(4):                                     # Chaikin: rounds every corner
            Q = [P[0]]
            for p0, p1 in zip(P[:-1], P[1:]):
                Q += [0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1]
            Q.append(P[-1])
            P = np.array(Q)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.P, self.z, self.v = P, z, speed

    def at(self, t):
        s = min(t * self.v, self.s[-1] - 1e-3)
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.P) - 2))
        f = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        p = self.P[i] + (self.P[i + 1] - self.P[i]) * f
        d = self.P[i + 1] - self.P[i]
        yaw = math.atan2(d[1], d[0])
        return np.array([p[0], p[1], self.z + 0.25 * math.sin(t * 1.3)]), yaw


def jackal_at(t, ground):
    y = JACKAL_Y0 + JACKAL_V * t
    x = JACKAL_X + 0.08 * math.sin(0.4 * t)
    return np.array([x, y, ground(x, y)]), math.pi / 2 + 0.04 * math.cos(0.4 * t)


def lerp(a, b, u):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * u


def ease(u):
    return armk.minjerk(min(max(u, 0.0), 1.0))


def make_shots(rover, drone, ground):
    """[(name, frames, cam(t, u) -> (eye, target, focal), caption, extra)] where t is film time (s)."""
    def establish(t, u):
        w = ease(u)
        return lerp((-95, -62, 46), (-70, -34, 20), w), lerp((-10, 4, 0), (-28, 0, 0), w), 24

    def chase(t, u):
        p, yaw = drone.at(t)
        back = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return p - 5.5 * back + 1.8 * side + [0, 0, 2.4], p + 4.0 * back - [0, 0, 1.8], 26

    def rover_wide(t, u):
        x = rover.at(t)[0]
        a = math.radians(-120 + 70 * u)
        c = np.array([x + 0.1, 0.55, ground(x, 0.0) + 0.35])
        return c + [3.3 * math.cos(a), 3.3 * math.sin(a), 1.5], c, 28

    def rover_close(t, u):
        x = rover.at(t)[0]
        g = ground(x, 0.0)
        return (np.array([x + 1.35 + 0.15 * u, -0.45, g + 1.10]), np.array([x + 0.10, 0.80, g + 0.25]), 30)

    def data(t, u):
        p, yaw = drone.at(t)
        return p + [-6.0, -6.0, 4.0], p, 30

    def maize(t, u):
        p, _ = jackal_at(t, ground)
        return p + [-1.9, -2.6 + 0.6 * u, 0.85], p + [1.3, 1.6, 0.55], 22

    def closing(t, u):
        x = rover.at(t)[0]
        w = ease(u)
        c = np.array([x, 0.5, ground(x, 0.0)])
        return c + lerp((-3.0, -4.0, 2.2), (-34.0, -42.0, 30.0), w), c + lerp((0, 0, 0.3), (8, 4, 0), w), 24

    return [
        ("establish", 210, establish, "Farm digital twin in NVIDIA Isaac Sim: lettuce beds, a maize block and "
                                      "three field robots", {}),
        ("chase", 330, chase, "Survey drone: its camera streams synthetic RGB and semantic labels", {"pip": True}),
        ("rover_wide", 300, rover_wide, "Weeding rover: a UR5e on a Clearpath Ridgeback pulls weeds between the "
                                        "lettuce", {}),
        ("rover_close", 270, rover_close, "Each weed is located, gripped, lifted and dropped in the bin on the deck",
         {}),
        ("data", 270, data, "Synthetic training data from the drone: RGB, depth, semantic segmentation, weed boxes",
         {"grid": True}),
        ("maize", 210, maize, "Clearpath Jackal scouting the maize block from the farm track", {}),
        ("closing", 210, closing, "Procedural crops, real HDRI sky, robots posed by forward and inverse kinematics",
         {}),
    ]


def plan(shots):
    """Frame index -> (shot index, film time s, fraction through the shot)."""
    out = []
    f0 = 0
    for i, s in enumerate(shots):
        for k in range(s[1]):
            out.append((i, (f0 + k) / FPS, k / max(1, s[1] - 1)))
        f0 += s[1]
    return out

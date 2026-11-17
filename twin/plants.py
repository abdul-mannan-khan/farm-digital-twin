"""Procedural crop plants (numpy only): lettuce, maize and weeds as textured leaf meshes, plus their leaf textures.

Every plant is a list of parts; a part is a triangle mesh (points, face indices, uvs) with the name of the texture
it uses. Leaves are parametric surfaces: u runs from the leaf base to its tip, v across the blade from one edge to
the other. Shape comes from botany rather than boxes: lettuce leaves are obovate, cupped and ruffled at the edge and
set in a rosette at the golden angle; maize leaves are long tapering straps with a folded midrib that rise from the
stalk nodes in two opposite ranks and arch over; weeds are flat, lobed rosettes.
"""
import math

import numpy as np

GOLDEN = math.radians(137.508)


# ------------------------------------------------------------------ mesh helpers
def grid_faces(nu, nv):
    """Two triangles per cell of an nu x nv vertex grid (row-major, u slow)."""
    i = np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None, :]
    a, b, c, d = i, i + 1, i + nv, i + nv + 1
    return np.stack([np.stack([a, c, b], -1), np.stack([b, c, d], -1)], 2).reshape(-1, 3)


def vertex_normals(P, F):
    """Area-weighted smooth normals (meshes without authored normals render faceted)."""
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def part(points, faces, uv, tex):
    return {"points": np.asarray(points, np.float32), "faces": np.asarray(faces, np.int32),
            "uv": np.asarray(uv, np.float32), "tex": tex}


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def merge(parts):
    """Merge parts that share a texture into one mesh each (fewer prims per plant)."""
    out = {}
    for p in parts:
        o = out.setdefault(p["tex"], {"points": [], "faces": [], "uv": [], "n": 0})
        o["points"].append(p["points"])
        o["faces"].append(p["faces"] + o["n"])
        o["uv"].append(p["uv"])
        o["n"] += len(p["points"])
    return [part(np.concatenate(o["points"]), np.concatenate(o["faces"]), np.concatenate(o["uv"]), t)
            for t, o in out.items()]


def sweep_leaf(spine, frame_y, frame_n, width, nv, fold=0.0, cup=0.0, ruffle=None, v_range=(-1.0, 1.0)):
    """Surface around a spine curve.

    spine: (nu, 3) points from base to tip; frame_y: (nu, 3) unit vectors across the blade; frame_n: (nu, 3) unit
    normals of the blade; width: (nu,) half-widths; fold: midrib V-fold depth (fraction of half-width, edges rise);
    cup: across-blade curvature (edges rise ~ v^2); ruffle: (nu, nv) extra normal offsets at the edge."""
    nu = len(spine)
    v = np.linspace(*v_range, nv)
    off_y = width[:, None] * v[None, :]
    off_n = width[:, None] * (fold * np.abs(v)[None, :] + cup * (v ** 2)[None, :])
    if ruffle is not None:
        off_n = off_n + ruffle
    P = spine[:, None, :] + frame_y[:, None, :] * off_y[..., None] + frame_n[:, None, :] * off_n[..., None]
    u = np.linspace(0, 1, nu)
    uv = np.stack(np.meshgrid((v - v_range[0]) / (v_range[1] - v_range[0]), u), -1).reshape(-1, 2)
    return P.reshape(-1, 3), grid_faces(nu, nv), uv


# ------------------------------------------------------------------ lettuce
def lettuce(rng, size=1.0, kind="green"):
    """Head lettuce: 16-24 leaves at the golden angle, inner ones small and upright, outer ones large and spreading."""
    n = int(rng.integers(16, 25))
    parts = []
    for k in range(n):
        f = k / (n - 1)                               # 0 innermost (youngest) .. 1 outermost
        L = size * (0.07 + 0.11 * f ** 0.9) * rng.uniform(0.9, 1.1)
        elev0 = math.radians(84 - 50 * f + rng.uniform(-5, 5))      # angle above the ground at the base
        # inner leaves curl inwards over the heart (negative bend), outer ones arch outwards and lie down
        bend = math.radians(-70 + 150 * f) if f < 0.55 else math.radians(25 + 45 * f)
        az = k * GOLDEN + rng.uniform(-0.15, 0.15)
        nu, nv = 16, 13
        u = np.linspace(0, 1, nu)
        th = elev0 - bend * u ** 1.5
        d = np.stack([np.cos(th), np.zeros(nu), np.sin(th)], 1)
        spine = np.concatenate([[[0, 0, 0]], np.cumsum(d[:-1] * (L / (nu - 1)), 0)])
        spine += [0.012 * size * (1 - f) * 0, 0, 0.004 * size]
        tang = d
        y = np.tile([0.0, 1.0, 0.0], (nu, 1))
        nrm = np.cross(tang, y)
        nrm *= -np.sign(nrm[:, 2:3] + 1e-9)                           # normals point up
        # obovate outline: narrow petiole, widest at 70% of the length, round tip
        w = size * (0.03 + 0.07 * f) * np.clip(np.sin(np.pi * np.clip(u * 0.85 + 0.08, 0, 1)) ** 0.8
                                                 * (0.35 + 0.65 * u ** 0.5), 0.02, None)
        w[-1] *= 0.55
        vv = np.linspace(-1, 1, nv)
        freq, ph = rng.uniform(3, 5), rng.uniform(0, 6.3)
        amp = size * (0.003 + 0.007 * f) * (1.5 if kind == "red" else 1.0)
        ruffle = amp * np.abs(vv)[None, :] ** 4 * np.sin(freq * 2 * np.pi * u[:, None] + ph + 1.2 * np.sign(vv)[None, :]) \
            * u[:, None] ** 0.7
        cup = 1.4 - 0.9 * f
        P, F, UV = sweep_leaf(spine, y, nrm, w, nv, cup=cup, ruffle=ruffle)
        P = P @ rot_z(az).T
        tex = f"lettuce_{kind}_{'inner' if f < 0.45 else 'outer'}"
        parts.append(part(P, F, UV, tex))
    return merge(parts)


# ------------------------------------------------------------------ maize
def tube(path, radius, nseg, tex, v_scale=1.0):
    """Tapered tube along a path (n, 3) with radius (n,)."""
    n = len(path)
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    a = np.cross(t, [1.0, 0, 0])
    a[np.linalg.norm(a, axis=1) < 1e-6] = [0, 1.0, 0]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(t, a)
    ang = np.linspace(0, 2 * np.pi, nseg + 1)
    P = path[:, None, :] + radius[:, None, None] * (np.cos(ang)[None, :, None] * a[:, None, :]
                                                    + np.sin(ang)[None, :, None] * b[:, None, :])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    uv = np.stack(np.meshgrid(ang / (2 * np.pi), s * v_scale), -1).reshape(-1, 2)
    return part(P.reshape(-1, 3), grid_faces(n, nseg + 1), uv, tex)


def maize(rng, height=1.9, stage=1.0):
    """Maize near tasseling: tapered stalk, 11-14 strap leaves in two opposite ranks, a tassel and one or two ears."""
    H = height * rng.uniform(0.9, 1.1) * stage
    parts = []
    lean = rng.normal(0, 0.03, 2)
    ns = 24
    z = np.linspace(0, H, ns)
    stalk = np.stack([lean[0] * (z / H) ** 2 * H, lean[1] * (z / H) ** 2 * H, z], 1)
    r = 0.017 * (1 - 0.65 * (z / H)) + 0.003
    parts.append(tube(stalk, r, 8, "maize_stalk", v_scale=2.0))
    nl = int(rng.integers(11, 15))
    rank = rng.uniform(0, np.pi)                    # the two leaf ranks lie in one vertical plane
    for k in range(nl):
        f = k / (nl - 1)
        hz = H * (0.04 + 0.86 * f ** 1.1)
        L = H * (0.25 + 0.38 * math.sin(math.pi * min(1.0, 0.15 + 0.95 * f)) ** 0.9) * rng.uniform(0.9, 1.1)
        if f > 0.85:
            L *= 0.7
        wmax = 0.05 + 0.035 * math.sin(math.pi * f)
        az = rank + (np.pi if k % 2 else 0.0) + rng.normal(0, 0.25)
        nu, nv = 22, 5
        u = np.linspace(0, 1, nu)
        a0 = math.radians(70 - 25 * f + rng.uniform(-8, 8))           # leaves leave the stalk steeply ...
        a1 = math.radians(-35 + 60 * f + rng.uniform(-10, 10))          # ... and droop towards the tip
        th = a0 + (a1 - a0) * u ** 1.3
        d = np.stack([np.cos(th), np.zeros(nu), np.sin(th)], 1)
        spine = np.concatenate([[[0, 0, 0]], np.cumsum(d[:-1] * (L / (nu - 1)), 0)])
        y = np.tile([0.0, 1.0, 0.0], (nu, 1))
        tw = rng.uniform(-0.6, 0.6) * u                                 # slight twist along the blade
        y = np.stack([np.zeros(nu), np.cos(tw), np.sin(tw)], 1)
        y -= d * np.sum(y * d, 1, keepdims=True)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        nrm = np.cross(d, y)
        nrm *= np.sign(nrm[:, 2:3] + 1e-9)
        w = wmax * np.clip(np.minimum(u / 0.12, 1.0) ** 0.5 * (1 - u) ** 0.65, 0.05, None) * 0.75 + 0.004 * (1 - u)
        vv = np.linspace(-1, 1, nv)
        wav = 0.006 * np.abs(vv)[None, :] ** 2 * np.sin(np.pi * 2 * 14 * u[:, None] + rng.uniform(0, 6))
        P, F, UV = sweep_leaf(spine, y, nrm, w, nv, fold=0.5, ruffle=wav * (u[:, None] > 0.2))
        P = P @ rot_z(az).T + stalk[min(ns - 1, int(hz / H * (ns - 1)))] + [0, 0, 0]
        P[:, 2] += hz - stalk[min(ns - 1, int(hz / H * (ns - 1))), 2]
        parts.append(part(P, F, UV, "maize_leaf_low" if f < 0.25 else "maize_leaf"))
    # tassel: a central spike and drooping side branches
    top = stalk[-1]
    for b in range(int(rng.integers(10, 16)) + 1):
        s = np.linspace(0, 1, 8)
        if b == 0:
            path = top + np.stack([0.02 * s ** 2, np.zeros(8), 0.22 * s], 1)
            rr = 0.006 * (1 - 0.6 * s)
        else:
            a = rng.uniform(0, 2 * np.pi)
            l_ = rng.uniform(0.10, 0.20)
            path = top + [0, 0, rng.uniform(0.02, 0.12)] + np.stack(
                [l_ * s * math.cos(a), l_ * s * math.sin(a), 0.07 * s - 0.16 * s ** 2], 1)
            rr = 0.0045 * (1 - 0.5 * s)
        parts.append(tube(path, rr, 5, "maize_tassel", v_scale=8.0))
    # ears in their husks at the upper-middle nodes, with silk
    for e in range(int(rng.integers(1, 3))):
        hz = H * (0.45 + 0.08 * e)
        a = rank + np.pi / 2 + rng.normal(0, 0.3) + np.pi * e
        s = np.linspace(0, 1, 10)
        base = stalk[int(hz / H * (ns - 1))]
        dirv = np.array([math.cos(a) * 0.45, math.sin(a) * 0.45, 0.9])
        dirv /= np.linalg.norm(dirv)
        path = base + 0.2 * s[:, None] * dirv
        rr = 0.024 * np.sin(np.pi * np.clip(s * 0.9 + 0.08, 0, 1)) ** 0.6 + 0.004
        parts.append(tube(path, rr, 8, "maize_husk", v_scale=4.0))
        tip = path[-1]
        for _ in range(5):
            sp = tip + np.stack([rng.normal(0, 0.012) * s, rng.normal(0, 0.012) * s, 0.04 * s - 0.03 * s ** 2], 1)
            parts.append(tube(sp, np.full(10, 0.0012), 3, "maize_silk"))
    return merge(parts)


# ------------------------------------------------------------------ weeds
def weed(rng, size=1.0):
    """Low weed rosette (dandelion-like): narrow lobed leaves lying almost flat."""
    n = int(rng.integers(7, 13))
    parts = []
    for k in range(n):
        L = size * rng.uniform(0.06, 0.12)
        nu, nv = 10, 5
        u = np.linspace(0, 1, nu)
        th = math.radians(rng.uniform(15, 35)) - math.radians(20) * u
        d = np.stack([np.cos(th), np.zeros(nu), np.sin(th)], 1)
        spine = np.concatenate([[[0, 0, 0]], np.cumsum(d[:-1] * (L / (nu - 1)), 0)])
        y = np.tile([0.0, 1.0, 0.0], (nu, 1))
        nrm = np.cross(d, y)
        nrm *= np.sign(nrm[:, 2:3] + 1e-9)
        lobes = 0.55 + 0.45 * np.abs(np.sin(u * np.pi * rng.uniform(3, 5)))     # toothed outline
        w = size * 0.016 * np.sin(np.pi * np.clip(u, 0.05, 0.95)) ** 0.7 * lobes
        P, F, UV = sweep_leaf(spine, y, nrm, w, nv, cup=0.3)
        parts.append(part(P @ rot_z(k * GOLDEN + rng.uniform(-0.2, 0.2)).T, F, UV, "weed"))
    return merge(parts)


# ------------------------------------------------------------------ textures
def _noise(rng, h, w, scales=(4, 16, 64), amps=(0.5, 0.3, 0.2)):
    """Cheap value noise: sum of bilinearly upsampled random grids."""
    from PIL import Image
    out = np.zeros((h, w), np.float32)
    for s, a in zip(scales, amps):
        g = rng.random((max(2, h * s // w), s)).astype(np.float32)
        out += a * np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                              np.float32) / 255.0
    return out


def leaf_texture(rng, kind, w=256, h=512):
    """RGB leaf texture: u (image y) base -> tip, v (image x) edge -> edge; midrib, veins, colour gradients."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = 1 - yy / (h - 1)                                 # image row 0 is the tip
    v = xx / (w - 1) * 2 - 1
    n = _noise(rng, h, w)
    if kind.startswith("maize"):
        base = np.array([0.12, 0.31, 0.05]) if kind == "maize_leaf" else np.array([0.20, 0.33, 0.07])
        col = base[None, None, :] * (0.85 + 0.3 * n[..., None])
        stripes = 0.5 + 0.5 * np.cos(np.abs(v) * np.pi * 22)
        col *= (0.93 + 0.07 * stripes)[..., None]
        mid = np.exp(-(v / 0.05) ** 2)
        col = col * (1 - 0.7 * mid[..., None]) + np.array([0.45, 0.56, 0.28]) * 0.7 * mid[..., None]
        if kind == "maize_leaf_low":                      # senescing tips on the lowest leaves
            dry = np.clip((u - 0.55) / 0.35 + 0.3 * (n - 0.5), 0, 1)[..., None]
            col = col * (1 - dry) + np.array([0.55, 0.45, 0.25]) * dry
        else:
            tipy = np.clip((u - 0.85) / 0.15, 0, 1)[..., None] * 0.35
            col = col * (1 - tipy) + np.array([0.45, 0.50, 0.20]) * tipy
    elif kind.startswith("lettuce"):
        inner = kind.endswith("inner")
        red = "_red_" in kind
        c0 = np.array([0.50, 0.66, 0.20]) if inner else np.array([0.22, 0.46, 0.08])
        c1 = np.array([0.34, 0.58, 0.12]) if inner else np.array([0.12, 0.32, 0.04])
        col = c0 * (1 - u[..., None]) + c1 * u[..., None]
        col = col * (0.9 + 0.2 * n[..., None])
        # midrib and branching lateral veins (pale), running out towards the edge and tip
        mid = np.exp(-(v / (0.06 * (1.2 - u))) ** 2)
        lat = np.abs(np.sin((u - 0.6 * np.abs(v)) * np.pi * 9))
        veins = np.clip(np.exp(-((1 - lat) / 0.035) ** 2) * (np.abs(v) < 0.85), 0, 1) * 0.18
        pale = np.array([0.72, 0.82, 0.52])
        k_ = np.clip(0.8 * mid + veins * (1 - u * 0.5), 0, 1)[..., None]
        col = col * (1 - k_) + pale * k_
        if red:                                       # anthocyanin: burgundy edges and tips
            rr = np.clip((np.abs(v) * 0.8 + u * 0.6 - 0.45) * 1.8 + 0.3 * (n - 0.5), 0, 1)[..., None]
            col = col * (1 - rr) + np.array([0.36, 0.08, 0.12]) * rr * (0.8 + 0.4 * n[..., None])
    elif kind == "weed":
        col = np.array([0.16, 0.33, 0.07]) * (0.8 + 0.4 * n[..., None])
        mid = np.exp(-(v / 0.07) ** 2)[..., None]
        col = col * (1 - mid) + np.array([0.45, 0.30, 0.25]) * mid
    elif kind == "maize_stalk":
        col = np.array([0.20, 0.36, 0.08]) * (0.85 + 0.3 * n[..., None])
        node = np.exp(-(((yy / h * 6) % 1 - 0.5) / 0.03) ** 2)[..., None]
        col = col * (1 - node * 0.4) + np.array([0.45, 0.40, 0.18]) * node * 0.4
    elif kind == "maize_tassel":
        col = np.array([0.62, 0.50, 0.24]) * (0.8 + 0.4 * n[..., None])
    elif kind == "maize_husk":
        col = np.array([0.42, 0.58, 0.20]) * (0.85 + 0.3 * n[..., None])
        col *= (0.9 + 0.1 * np.cos(v * np.pi * 10))[..., None]
    elif kind == "maize_silk":
        col = np.array([0.55, 0.38, 0.20]) * (0.8 + 0.4 * n[..., None])
    else:
        raise KeyError(kind)
    return (np.clip(col, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)   # stored sRGB


TEXTURES = ("lettuce_green_inner", "lettuce_green_outer", "lettuce_red_inner", "lettuce_red_outer",
            "maize_leaf", "maize_leaf_low", "maize_stalk", "maize_tassel", "maize_husk", "maize_silk", "weed")


def write_textures(folder, seed=0):
    import os

    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for t in TEXTURES:
        Image.fromarray(leaf_texture(rng, t)).save(os.path.join(folder, f"{t}.png"))
    return {t: os.path.join(folder, f"{t}.png") for t in TEXTURES}


def to_obj(parts, path):
    """Debug export."""
    with open(path, "w") as f:
        n = 1
        for p in parts:
            f.write(f"o {p['tex']}\n")
            for q in p["points"]:
                f.write("v %.4f %.4f %.4f\n" % tuple(q))
            for t in p["faces"]:
                f.write("f %d %d %d\n" % tuple(t + n))
            n += len(p["points"])

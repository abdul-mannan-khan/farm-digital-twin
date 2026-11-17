"""Farm digital twin in NVIDIA Isaac Sim: a lettuce and maize field with a weeding rover and a survey drone.

The field is built from scratch: a terrain mesh shaped into raised lettuce beds, maize ridges and a farm track;
~30k procedurally modelled crop plants (plants.py) placed by point instancers; NVIDIA vegetation for the margins
and hedgerows; a real HDRI sky. Robots: a Clearpath Ridgeback carrying a UR5e with a Robotiq gripper weeds the
lettuce beds, a quadcopter flies a survey, a Clearpath Jackal scouts the maize rows. Replicator renders the film
and the drone's synthetic data (RGB, depth, semantic labels).

  /isaac-sim/python.sh farm.py --assets /root/assets --out frames --preview
"""
import argparse
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--assets", default="/root/assets")
ap.add_argument("--out", default="frames")
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--preview", action="store_true", help="render the preview cameras only")
ap.add_argument("--frames", default="", help="comma-separated frame numbers")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--subframes", type=int, default=8)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--density", type=float, default=1.0, help="plant density scale (for quick tests)")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

from isaacsim import SimulationApp                      # noqa: E402
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": W, "height": H})

import carb                                            # noqa: E402
import numpy as np                                     # noqa: E402
import omni.replicator.core as rep                     # noqa: E402
import omni.usd                                        # noqa: E402
from PIL import Image                                  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm as armk                                     # noqa: E402
import plants                                          # noqa: E402

ROOT = get_assets_root_path()
VEG = f"{ROOT}/NVIDIA/Assets/Vegetation"
A = args.assets
rng = np.random.default_rng(args.seed)

omni.usd.get_context().new_stage()
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

settings = carb.settings.get_settings()
for k, v in {"/rtx/indirectDiffuse/enabled": True, "/rtx/ambientOcclusion/enabled": True,
             "/rtx/reflections/enabled": True, "/rtx/directLighting/sampledLighting/enabled": True,
             "/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 2,
             "/rtx/post/histogram/enabled": os.environ.get("AUTOEXP", "1") == "1",
             "/rtx/shadows/enabled": True, "/rtx/translucency/enabled": True}.items():
    settings.set(k, v)


# ------------------------------------------------------------------ materials
_mats = {}


def pbr(name, diffuse=None, normal=None, rough_tex=None, color=(0.5, 0.5, 0.5), rough=0.7, metal=0.0,
        scale=(1.0, 1.0), emit=None, tint=None):
    """OmniPBR material; textures are file paths (normal and roughness maps read as raw data)."""
    if name in _mats:
        return _mats[name]
    path = f"/World/Looks/{name}"
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/Shader")
    s.SetSourceAsset("OmniPBR.mdl", "mdl")
    s.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    s.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if tint is not None:                            # with a texture the colour constant is ignored; tint instead
        s.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    s.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metal)
    s.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*scale))
    if diffuse:
        s.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(diffuse)
    if normal:
        i = s.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset)
        i.Set(normal)
        i.GetAttr().SetColorSpace("raw")
    if rough_tex:
        i = s.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
        i.Set(rough_tex)
        i.GetAttr().SetColorSpace("raw")
        s.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    if emit:
        s.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        s.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit[0]))
        s.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(emit[1]))
    m.CreateSurfaceOutput("mdl").ConnectToSource(s.ConnectableAPI(), "out")
    m.CreateDisplacementOutput("mdl").ConnectToSource(s.ConnectableAPI(), "out")
    m.CreateVolumeOutput("mdl").ConnectToSource(s.ConnectableAPI(), "out")
    _mats[name] = m
    return m


def bind(prim, m):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.strongerThanDescendants)


def tex(name, kind, res="2k"):
    """Poly Haven texture file (downloaded by get_assets.sh)."""
    ext = "exr" if kind == "hdri" else "jpg"
    return f"{A}/{name}_{kind}_{res}.{ext}"


def soil_mat(name, ph, scale, tint=(1, 1, 1), rough=None):
    return pbr(name, diffuse=tex(ph, "diff"), normal=tex(ph, "nor_gl"), rough_tex=tex(ph, "rough"),
               tint=tint, rough=rough or 0.9, scale=scale)


def label(prim, cls):
    """Semantic class for Replicator's segmentation."""
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[cls], instance_name="class")
    except Exception:                                  # older API
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, cls)


# ------------------------------------------------------------------ layout (metres; x east, y north, z up)
LET_X = (-44.0, -2.0)            # lettuce block, beds run along x
MAI_X = (3.0, 44.0)              # maize block, rows run along x
FIELD_Y = (-28.0, 28.0)
TRAM = 0.60                      # half width of the tramline at y = 0 where the weeding rover drives
BED_HALF, BED_SHOULDER, BED_H, BED_PITCH = 0.50, 0.18, 0.14, 1.60
LET_ROWS, LET_SPACING = (-0.30, 0.0, 0.30), 0.30
MAI_PITCH, MAI_SPACING = 0.76, 0.18
TRACK_X = (-1.2, 2.0)


def _beds():
    c, out = TRAM + 0.10 + BED_HALF + BED_SHOULDER * 0.5, []
    while c + BED_HALF < FIELD_Y[1]:
        out += [c, -c]
        c += BED_PITCH
    return sorted(out)


BEDS = np.array(_beds())
ROW1 = float(BEDS[BEDS > 0].min() + LET_ROWS[0])     # first lettuce row north of the tramline
TARGET_X = (-36.1, -34.6, -33.2, -31.6, -30.1, -28.5, -27.0, -25.4, -23.9, -22.3, -20.8, -19.2)
TARGETS = np.array([[x, ROW1 + 0.015 * math.sin(3.1 * x)] for x in TARGET_X])


def smooth_noise(x, y, seed=0):
    """Gentle field undulation (a few long sine waves; deterministic, vectorised)."""
    r = np.random.default_rng(seed)
    h = np.zeros_like(x)
    for k in range(6):
        a, ph = r.uniform(0, np.pi), r.uniform(0, 2 * np.pi)
        lam = r.uniform(25, 90)
        h += np.sin((x * np.cos(a) + y * np.sin(a)) * 2 * np.pi / lam + ph) / (k + 2)
    return 0.18 * h


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def height(x, y):
    """Ground height: undulation + lettuce beds / maize ridges / wheel ruts of the farm track."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = smooth_noise(x, y)
    inlet = smoothstep(LET_X[0] - 1, LET_X[0], x) * (1 - smoothstep(LET_X[1], LET_X[1] + 0.5, x))
    d = np.min(np.abs(y[..., None] - BEDS), axis=-1)
    bed = BED_H * (1 - smoothstep(BED_HALF, BED_HALF + BED_SHOULDER, d))
    h = h + inlet * bed * (np.abs(y) < FIELD_Y[1])
    inmai = smoothstep(MAI_X[0] - 0.8, MAI_X[0], x) * (1 - smoothstep(MAI_X[1], MAI_X[1] + 0.8, x))
    ridge = 0.07 * (0.5 + 0.5 * np.cos(2 * np.pi * y / MAI_PITCH)) ** 2
    h = h + inmai * ridge * (np.abs(y) < FIELD_Y[1])
    tr = (x > TRACK_X[0]) & (x < TRACK_X[1])
    cx = (TRACK_X[0] + TRACK_X[1]) / 2
    rut = 0.04 * (np.exp(-((x - cx - 0.8) / 0.18) ** 2) + np.exp(-((x - cx + 0.8) / 0.18) ** 2))
    return h - tr * rut


def ground_z(x, y):
    return float(height(np.array([x]), np.array([y]))[0])


def mesh(path, P, F, UV=None, double=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(double)
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(
        plants.vertex_normals(np.asarray(P, np.float64), np.asarray(F)), np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if UV is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(UV, np.float32)))
    return m


def terrain():
    """Field terrain at 8 cm across the beds, plus a large low-resolution meadow around it."""
    fine = [np.arange(e - 1.0, e + 1.0, 0.05) for e in (LET_X[0], LET_X[1], MAI_X[0], MAI_X[1])]
    xs = np.unique(np.round(np.concatenate([np.arange(-50.0, 50.0 + 1e-6, 0.4)] + fine), 4))
    ys = np.arange(-32.0, 32.0 + 1e-6, 0.08)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = height(X, Y)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = plants.grid_faces(len(xs), len(ys))
    F = F[:, [0, 2, 1]]                                   # faces up
    m = mesh("/World/ground/field", P, F, P[:, :2] / 2.0)
    # materials by face: track, field soil, grass headland around the field
    c = P[F].mean(1)
    track = (c[:, 0] > TRACK_X[0]) & (c[:, 0] < TRACK_X[1])
    infield = (np.abs(c[:, 1]) < FIELD_Y[1] + 0.6) & (((c[:, 0] > LET_X[0] - 1) & (c[:, 0] < LET_X[1] + 0.5))
                                                       | ((c[:, 0] > MAI_X[0] - 0.8) & (c[:, 0] < MAI_X[1] + 0.8)))
    soil = infield & ~track
    head = ~infield & ~track
    for nm, sel, mat in (("soil", soil, M["soil"]), ("track", track, M["track"]), ("headland", head, M["grass"])):
        sub = UsdGeom.Subset.CreateGeomSubset(m, nm, UsdGeom.Tokens.face, Vt.IntArray.FromNumpy(
            np.nonzero(sel)[0].astype(np.int32)), familyName="materialBind")
        bind(sub.GetPrim(), mat)
    label(m.GetPrim(), "soil")
    # meadow beyond: follows the field's own undulation a few cm lower (so it continues seamlessly under the field
    # edge and never pokes through), rising into gentle hills far away
    g = np.concatenate([np.linspace(-700, -60, 33)[:-1], np.arange(-60, 60.01, 4.0), np.linspace(60, 700, 33)[1:]])
    X, Y = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(X, Y)
    Z = smooth_noise(X, Y) - 0.08 + smoothstep(150, 500, r) * (smooth_noise(X, Y, 3) * 10 + 4)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = plants.grid_faces(len(g), len(g))[:, [0, 2, 1]]
    mm = mesh("/World/ground/meadow", P, F, P[:, :2] / 6.0)
    bind(mm.GetPrim(), M["meadow"])
    label(mm.GetPrim(), "grass")


# ------------------------------------------------------------------ plants
PROTO_SEEDS = {"lettuce_green": 5, "lettuce_red": 3, "maize": 6, "weed": 3}


def build_protos(inst, kind, n, maker):
    """Prototype prims under a point instancer; returns their paths."""
    paths = []
    r = np.random.default_rng(hash(kind) % 2 ** 32)
    for i in range(n):
        p = f"{inst}/protos/{kind}{i}"
        UsdGeom.Xform.Define(stage, p)
        for j, prt in enumerate(maker(r)):
            mm = mesh(f"{p}/m{j}", prt["points"], prt["faces"], prt["uv"], double=True)
            bind(mm.GetPrim(), LEAF[prt["tex"]])
        label(stage.GetPrimAtPath(p), kind.split("_")[0])
        paths.append(p)
    return paths


def instancer(path, protos, idx, pos, yaw, scale):
    pi = UsdGeom.PointInstancer.Define(stage, path)
    pi.CreatePrototypesRel().SetTargets([Sdf.Path(p) for p in protos])
    pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(idx, np.int32)))
    pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pos, np.float32)))
    h = np.asarray(yaw) / 2
    q = np.stack([np.cos(h), np.zeros_like(h), np.zeros_like(h), np.sin(h)], 1)
    pi.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(a), float(b), float(c), float(d)) for a, b, c, d in q]))
    pi.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.repeat(np.asarray(scale, np.float32)[:, None], 3, 1)))
    return pi


def plant_field():
    dens = args.density
    # lettuce: three rows per bed; every third bed red lettuce
    pos, idx, yaw, sc = [], [], [], []
    lp = build_protos("/World/crops/lettuce", "lettuce_green", PROTO_SEEDS["lettuce_green"],
                      lambda r: plants.lettuce(r, size=1.0, kind="green"))
    lp += build_protos("/World/crops/lettuce", "lettuce_red", PROTO_SEEDS["lettuce_red"],
                       lambda r: plants.lettuce(r, size=1.0, kind="red"))
    ng = PROTO_SEEDS["lettuce_green"]
    for b, yc in enumerate(BEDS):
        red = (int(round((abs(yc) - BEDS[BEDS > 0].min()) / BED_PITCH)) % 3 == 2)
        for ry in LET_ROWS:
            xs = np.arange(LET_X[0] + 0.4, LET_X[1] - 0.3, LET_SPACING / dens)
            xs = xs + rng.normal(0, 0.025, len(xs)) + (0.15 if ry == 0.0 else 0.0)
            keep = rng.random(len(xs)) > 0.03                         # a few gaps, as in a real planting
            ys = yc + ry + rng.normal(0, 0.02, len(xs))
            # the weeds the rover pulls grow in gaps of the first row next to the tramline
            keep &= np.min(np.hypot(xs[:, None] - TARGETS[None, :, 0], ys[:, None] - TARGETS[None, :, 1]), 1) > 0.22
            xs, ys = xs[keep], ys[keep]
            pos.append(np.stack([xs, ys, height(xs, ys)], 1))
            idx.append(ng + rng.integers(0, PROTO_SEEDS["lettuce_red"], len(xs)) if red
                       else rng.integers(0, ng, len(xs)))
            yaw.append(rng.uniform(0, 2 * np.pi, len(xs)))
            sc.append(rng.uniform(0.85, 1.15, len(xs)) * (0.95 + 0.1 * (b % 2)))
    pos = np.concatenate(pos)
    instancer("/World/crops/lettuce", lp, np.concatenate(idx), pos, np.concatenate(yaw), np.concatenate(sc))
    n_let = len(pos)
    # maize: one plant row per ridge
    mp = build_protos("/World/crops/maize", "maize", PROTO_SEEDS["maize"], lambda r: plants.maize(r, height=2.0))
    pos, idx, yaw, sc = [], [], [], []
    for yr in np.arange(-27.74, 27.8, MAI_PITCH):
        xs = np.arange(MAI_X[0] + 0.3, MAI_X[1] - 0.3, MAI_SPACING / dens) + rng.normal(0, 0.03, 1)
        xs = xs + rng.normal(0, 0.025, len(xs))
        xs = xs[rng.random(len(xs)) > 0.04]
        ys = yr + rng.normal(0, 0.02, len(xs))
        pos.append(np.stack([xs, ys, height(xs, ys) - 0.02], 1))
        idx.append(rng.integers(0, len(mp), len(xs)))
        yaw.append(np.pi / 2 + rng.normal(0, 0.35, len(xs)))      # leaves spread across the row, like real maize
        sc.append(rng.uniform(0.88, 1.12, len(xs)))
    pos = np.concatenate(pos)
    instancer("/World/crops/maize", mp, np.concatenate(idx), pos, np.concatenate(yaw), np.concatenate(sc))
    n_mai = len(pos)
    # weeds scattered in the lettuce beds and furrows
    wp = build_protos("/World/crops/weeds", "weed", PROTO_SEEDS["weed"], lambda r: plants.weed(r, size=1.2))
    nw = int(1800 * dens)
    xs = rng.uniform(LET_X[0] + 0.5, LET_X[1] - 0.5, nw)
    ys = rng.uniform(FIELD_Y[0] + 1, FIELD_Y[1] - 1, nw)
    ok = np.abs(ys) > TRAM + 0.15
    xs, ys = xs[ok], ys[ok]
    instancer("/World/crops/weeds", wp, rng.integers(0, len(wp), len(xs)),
              np.stack([xs, ys, height(xs, ys)], 1), rng.uniform(0, 2 * np.pi, len(xs)), rng.uniform(0.7, 1.3, len(xs)))
    global WEED_POS
    WEED_POS = np.concatenate([np.stack([xs, ys, height(xs, ys)], 1),
                               np.stack([TARGETS[:, 0], TARGETS[:, 1], height(TARGETS[:, 0], TARGETS[:, 1])], 1)])
    print(f"plants: {n_let} lettuce, {n_mai} maize, {len(xs)} weeds", flush=True)


def reference(path, url, pos=(0, 0, 0), yaw=0.0, scale=1.0):
    x = UsdGeom.Xform.Define(stage, path)
    t_ = x.AddTranslateOp()
    r_ = x.AddRotateZOp()
    s_ = x.AddScaleOp()
    t_.Set(Gf.Vec3d(*map(float, pos)))
    r_.Set(float(yaw))
    s_.Set(Gf.Vec3f(scale, scale, scale))
    c = stage.DefinePrim(path + "/asset", "Xform")
    c.GetReferences().AddReference(url)
    return x, t_, r_


def margins():
    """Grass headland, hedgerow shrubs and trees around the field (NVIDIA vegetation assets)."""
    grass = ["Grass_Short_A", "Grass_Short_B", "Grass_Short_C", "Fountain_Grass_Short", "Prairie_Dropseed"]
    protos = []
    for g in grass:
        p = f"/World/margin/grass/protos/{g}"
        x = UsdGeom.Xform.Define(stage, p)
        x.GetPrim().GetReferences().AddReference(f"{VEG}/Shrub/{g}.usd")
        protos.append(p)
    pts = []
    for _ in range(9000):
        side = rng.integers(0, 4)
        if side < 2:
            x = rng.uniform(-49, 49)
            y = (FIELD_Y[1] + rng.uniform(0.8, 3.8)) * (1 if side == 0 else -1)
        else:
            y = rng.uniform(-31, 31)
            x = rng.uniform(LET_X[0] - 5.5, LET_X[0] - 1.3) if side == 2 else rng.uniform(MAI_X[1] + 1.0, 49.5)
        pts.append((x, y))
    pts = np.array(pts)
    instancer("/World/margin/grass", protos, rng.integers(0, len(protos), len(pts)),
              np.stack([pts[:, 0], pts[:, 1], height(pts[:, 0], pts[:, 1])], 1),
              rng.uniform(0, 2 * np.pi, len(pts)), rng.uniform(0.4, 0.9, len(pts)) * 0.01)
    # hedgerow: shrubs and trees along the north and west edges, scattered trees beyond
    trees = ["American_Beech", "Elm_Sapling", "Fraxinus", "Honey_Locust", "Largetooth_Aspen", "Lombardy_Poplar",
             "Norway_Spruce", "Red_Ash", "Scarlet_Oak", "Shumard_Oak", "Gray_Birch"]
    shrubs = ["Privet", "Boxwood", "Holly", "Yew", "Forsythia", "Hawthorn"]
    k = 0
    for x in np.arange(-56, 58, 5.5):
        for y0 in (36.0, -37.0):
            y = y0 + rng.normal(0, 1.2)
            t = trees[rng.integers(0, len(trees))]
            reference(f"/World/margin/tree{k}", f"{VEG}/Trees/{t}.usd", (x + rng.normal(0, 1.5), y, ground_z(x, y)),
                      rng.uniform(0, 360), 0.01 * rng.uniform(0.85, 1.25))
            k += 1
            for _ in range(2):
                s = shrubs[rng.integers(0, len(shrubs))]
                sx, sy = x + rng.uniform(-2.5, 2.5), y0 - 2.0 + rng.normal(0, 0.6)
                folder = "Trees" if s == "Hawthorn" else "Shrub"
                reference(f"/World/margin/shrub{k}", f"{VEG}/{folder}/{s}.usd", (sx, sy, ground_z(sx, sy)),
                          rng.uniform(0, 360), 0.01 * rng.uniform(1.5, 2.4))
                k += 1
    for y in np.arange(-34, 36, 6.0):
        x = -58 + rng.normal(0, 1.5)
        t = trees[rng.integers(0, len(trees))]
        reference(f"/World/margin/tree{k}", f"{VEG}/Trees/{t}.usd", (x, y, ground_z(x, y)), rng.uniform(0, 360),
                  0.01 * rng.uniform(0.85, 1.25))
        k += 1


# ------------------------------------------------------------------ robots
RB = f"{ROOT}/Isaac/Robots"
A_RIDGE = f"{RB}/Clearpath/RidgebackUr/ridgeback_ur5.usd"
A_UR5E = f"{RB}/UniversalRobots/ur5e/ur5e.usd"
A_JACKAL = f"{RB}/Clearpath/Jackal/jackal.usd"
X500 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "x500.npz")   # tools/convert_x500.py
PROP_STEP = 47.0                             # degrees per frame: reads as a fast spin at 30 fps without strobing
ARM_MOUNT = np.array([0.12, 0.34, 0.38])     # UR5e base on the rover (rover frame: x forward, y left = bed side)
BIN = np.array([-0.24, -0.02, 0.28])         # weed bin on the rover deck (bin floor centre, rover frame)


def strip_physics(root):
    """Robots are posed kinematically from the timeline: switch off their rigid bodies, joints and articulations
    so the physics engine never moves them."""
    from pxr import UsdPhysics
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
            n += 1
        elif p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            n += 1
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print("physics off:", root, n, flush=True)


def prim_box(path, size, pos, mat):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    bind(c.GetPrim(), mat)
    return c


def weed_bin(root):
    """Open-top bin made of five thin panels."""
    L, Wd, Hh, t = 0.36, 0.30, 0.10, 0.008
    x, y, z = BIN
    prim_box(f"{root}/bin_floor", (L, Wd, t), (x, y, z + t / 2), M["bin"])
    prim_box(f"{root}/bin_n", (L, t, Hh), (x, y + Wd / 2, z + Hh / 2), M["bin"])
    prim_box(f"{root}/bin_s", (L, t, Hh), (x, y - Wd / 2, z + Hh / 2), M["bin"])
    prim_box(f"{root}/bin_e", (t, Wd, Hh), (x + L / 2, y, z + Hh / 2), M["bin"])
    prim_box(f"{root}/bin_w", (t, Wd, Hh), (x - L / 2, y, z + Hh / 2), M["bin"])


def build_rover():
    """Clearpath Ridgeback (its own UR5 removed) carrying a UR5e + Robotiq 2F-85 on a pedestal and a weed bin."""
    x, t_, r_ = reference("/World/rover", A_RIDGE)
    for p in stage.GetPrimAtPath("/World/rover/asset").GetChildren():
        if p.GetName().startswith("ur_arm"):
            p.SetActive(False)
    strip_physics("/World/rover")
    a = UsdGeom.Xform.Define(stage, "/World/rover/arm")
    a.AddTranslateOp().Set(Gf.Vec3d(*map(float, ARM_MOUNT)))
    ped = UsdGeom.Cylinder.Define(stage, "/World/rover/pedestal")
    ped.CreateRadiusAttr(0.075)
    ped.CreateHeightAttr(float(ARM_MOUNT[2] - 0.28))
    ped.AddTranslateOp().Set(Gf.Vec3d(float(ARM_MOUNT[0]), float(ARM_MOUNT[1]), float((ARM_MOUNT[2] + 0.28) / 2)))
    bind(ped.GetPrim(), M["pedestal"])
    asset = stage.DefinePrim("/World/rover/arm/asset", "Xform")
    asset.GetReferences().AddReference(A_UR5E)
    vs = asset.GetVariantSets()
    for k_, v in (("Gripper", "Robotiq_2f_85"), ("Physics", "None"), ("Sensor", "None")):
        if vs.HasVariantSet(k_):
            vs.GetVariantSet(k_).SetVariantSelection(v)
    weed_bin("/World/rover")
    # a camera mast at the front: the weeding camera looks down at the bed
    prim_box("/World/rover/mast", (0.03, 0.03, 0.55), (0.42, -0.30, 0.28 + 0.275), M["pedestal"])
    prim_box("/World/rover/camera", (0.06, 0.16, 0.05), (0.44, -0.30, 0.85), M["camera"])
    label(stage.GetPrimAtPath("/World/rover"), "robot")
    return {"t": t_, "r": r_, "root": "/World/rover/arm"}


def build_targets():
    """The weeds the rover pulls: separate prims so they can be lifted, carried and dropped in the bin."""
    out = []
    r = np.random.default_rng(11)
    for i, (x, y) in enumerate(TARGETS):
        p = f"/World/targets/w{i}"
        xf = UsdGeom.Xform.Define(stage, p)
        tr = xf.AddTranslateOp()
        ro = xf.AddOrientOp()
        sc = xf.AddScaleOp()
        tr.Set(Gf.Vec3d(float(x), float(y), ground_z(x, y)))
        ro.Set(Gf.Quatf(1, 0, 0, 0))
        sc.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        for j, prt in enumerate(plants.weed(r, size=1.3)):
            mm = mesh(f"{p}/m{j}", prt["points"], prt["faces"], prt["uv"], double=True)
            bind(mm.GetPrim(), LEAF[prt["tex"]])
        label(xf.GetPrim(), "weed")
        out.append({"prim": xf.GetPrim(), "t": tr, "o": ro, "s": sc, "home": (x, y, ground_z(x, y))})
    return out


def build_drone():
    """Holybro X500 (PX4's model, converted by tools/convert_x500.py): carbon frame, landing gear, four motors and
    propellers; each rotor is its own xform on the motor axis so the propellers spin."""
    d = np.load(X500)
    x = UsdGeom.Xform.Define(stage, "/World/drone")
    t_, r_ = x.AddTranslateOp(), x.AddRotateZOp()
    spins = []
    for i, pos in enumerate(d["rotor_pos"]):
        rx = UsdGeom.Xform.Define(stage, f"/World/drone/rotor{i}")
        rx.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        spins.append((rx.AddRotateZOp(), int(d["rotor_spin"][i])))
    for k, name in enumerate(d["names"]):
        r = int(d["rotor"][k])
        path = f"/World/drone/{name}" if r < 0 else f"/World/drone/rotor{r}/{name}"
        mm = mesh(path, d[f"v{k}"], d[f"f{k}"])
        bind(mm.GetPrim(), pbr(f"x500_{name}", color=tuple(map(float, d["rgb"][k])), rough=float(d["rough"][k]),
                               metal=float(d["metal"][k])))
    prim_box("/World/drone/camera", (0.05, 0.07, 0.05), (0.06, 0.0, -0.045), M["camera"])   # survey camera
    label(stage.GetPrimAtPath("/World/drone"), "robot")
    return {"t": t_, "r": r_, "spins": spins}


def build_jackal():
    x, t_, r_ = reference("/World/jackal", A_JACKAL)
    strip_physics("/World/jackal")
    label(stage.GetPrimAtPath("/World/jackal"), "robot")
    return {"t": t_, "r": r_}


_bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def bbox(prim):
    _bbox_cache.Clear()
    return _bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()


def wheel_spinners(root, pick, axis_world=(0.0, 1.0, 0.0)):
    """Wheels that roll: for each wheel prim chosen by pick(prim, size) (built while the robot sits at the origin
    facing +x), remember its centre and axle in its parent's frame, so a rotation by angle a about the axle through
    the centre can be written into its own translate/orient ops each frame."""
    out = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Xformable):
            continue
        b = bbox(p)
        if b.IsEmpty() or not pick(p, b.GetSize()):
            continue
        if p.IsInstanceProxy():                                # make the instance editable
            a = p
            while a and not a.IsInstance():
                a = a.GetParent()
            a.SetInstanceable(False)
            p = stage.GetPrimAtPath(p.GetPath())
        t, o = p.GetAttribute("xformOp:translate"), p.GetAttribute("xformOp:orient")
        if not (t.IsValid() and o.IsValid()):
            continue
        P2W = np.array(UsdGeom.Xformable(p).ComputeParentToWorldTransform(Usd.TimeCode.Default())).T
        W2P = np.linalg.inv(P2W)
        c = (W2P @ np.append(np.array(b.GetMidpoint()), 1.0))[:3]
        ax = W2P[:3, :3] @ np.asarray(axis_world, float)
        ax /= np.linalg.norm(ax)
        t0 = np.array(t.Get(), float)
        q0 = o.Get()
        out.append({"prim": p, "c": c, "ax": ax, "t0": t0, "q0": np.array([q0.GetReal(), *q0.GetImaginary()]),
                    "r": float(b.GetSize()[2]) / 2})
    print("wheels found under", root, len(out), flush=True)
    return out


def spin_wheels(wheels, dist):
    """Rotate every wheel by the angle that rolls it `dist` metres."""
    for w in wheels:
        a = dist / w["r"]
        u = w["ax"]
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
        t = w["c"] + R @ (w["t0"] - w["c"])
        qr = np.array([math.cos(a / 2), *(math.sin(a / 2) * u)])
        q0 = w["q0"]
        q = np.array([qr[0] * q0[0] - qr[1:] @ q0[1:], *(qr[0] * q0[1:] + q0[0] * qr[1:] + np.cross(qr[1:], q0[1:]))])
        T = np.eye(4)
        T[:3, :3] = armk.qmat(q)
        T[:3, 3] = t
        _set_pose(w["prim"], T)


def _set_pose(prim, T):
    """Write a 4x4 pose into a prim's translate/orient ops, keeping the attribute precision it was authored with."""
    t, o = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient")
    p, q = T[:3, 3], armk.mquat(T[:3, :3])
    t.Set(Gf.Vec3f(*map(float, p)) if "3f" in str(t.GetTypeName()) else Gf.Vec3d(*map(float, p)))
    if "quatf" in str(o.GetTypeName()):
        o.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
    else:
        o.Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))


def pose_arm(Ar, q, a):
    if "prims" not in Ar:
        r = Ar["root"] + "/asset"
        Ar["prims"] = {k: stage.GetPrimAtPath(f"{r}/{k}") for k in armk.LINKS + ["Gripper"]}
        Ar["fprims"] = {k: stage.GetPrimAtPath(f"{r}/Gripper/Robotiq_2F_85/{k}") for k in armk.fingers(0.0)}
        miss = [k for k, p in list(Ar["prims"].items()) + list(Ar["fprims"].items())
                if not p.IsValid() or not p.GetAttribute("xformOp:orient").IsValid()]
        print("arm missing prims:", miss or "none", flush=True)
        Ar["prims"] = {k: p for k, p in Ar["prims"].items() if k not in miss}
        Ar["fprims"] = {k: p for k, p in Ar["fprims"].items() if k not in miss}
    F = armk.fk(q)
    for k, p in Ar["prims"].items():
        _set_pose(p, F["gripper" if k == "Gripper" else k])
    for k, T in armk.fingers(a).items():
        if k in Ar["fprims"]:
            _set_pose(Ar["fprims"][k], T)


# ------------------------------------------------------------------ build
M = {"soil": soil_mat("soil", os.environ.get("SOIL", "brown_mud_02"), (1.0, 1.0)),
     "track": soil_mat("track", os.environ.get("TRACK", "brown_mud_dry"), (1.0, 1.0)),
     "grass": soil_mat("grass", os.environ.get("GRASS", "sparse_grass"), (1.0, 1.0), tint=(0.85, 0.95, 0.75)),
     "meadow": soil_mat("meadow", os.environ.get("MEADOW", "sparse_grass"), (1.0, 1.0),
                       tint=(0.45, 0.72, 0.30)),
     "bin": pbr("bin", color=(0.95, 0.42, 0.05), rough=0.5),
     "pedestal": pbr("pedestal", color=(0.55, 0.57, 0.6), rough=0.35, metal=0.8),
     "camera": pbr("camera", color=(0.05, 0.05, 0.06), rough=0.3)}
TEX = plants.write_textures(os.path.join(A, "leaves"), seed=args.seed)
LEAF = {t: pbr("leaf_" + t, diffuse=p, rough=0.45 if "lettuce" in t else 0.55) for t, p in TEX.items()}

t0 = time.time()
terrain()
plant_field()
margins()
ROVER, WEEDS, DRONE, JACKAL = build_rover(), build_targets(), build_drone(), build_jackal()
for _ in range(30):
    app.update()
# Ridgeback: its four wheels are meshes in the body (about 15 cm across, 8 cm wide); Jackal: its *_wheel_link prims
ROVER["wheels"] = wheel_spinners("/World/rover/asset", lambda p, s: abs(s[0] - s[2]) < 0.01 and 0.12 < s[0] < 0.2
                                 and s[1] < 0.1 and p.GetTypeName() == "Mesh")
JACKAL["wheels"] = wheel_spinners("/World/jackal/asset", lambda p, s: p.GetName().endswith("wheel_link"))
print(f"scene built in {time.time() - t0:.0f} s", flush=True)

# sky: real HDRI of a farm field, plus a sun for crisp shadows
dome = UsdLux.DomeLight.Define(stage, "/World/sky")
dome.CreateTextureFileAttr(tex(os.environ.get("HDRI", "farm_field"), "hdri", "4k"))
dome.CreateTextureFormatAttr("latlong")
dome.CreateIntensityAttr(float(os.environ.get("DOME", 1000.0)))
UsdGeom.Xformable(dome).AddRotateXYZOp().Set(Gf.Vec3f(float(os.environ.get("DOME_RX", 0)), 0,
                                                      float(os.environ.get("DOME_RZ", 0))))
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(float(os.environ.get("SUN", 3000)))
sun.CreateAngleAttr(0.6)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(float(os.environ.get("SUN_RX", 52)), 0,
                                                     float(os.environ.get("SUN_RZ", -40))))

for _ in range(200):                                    # stream the referenced vegetation
    app.update()

# ------------------------------------------------------------------ camera
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
cam_op = cam.AddTransformOp()


def set_camera(eye, tgt, focal):
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
    cam_op.Set(view.GetInverse())
    cam.GetFocalLengthAttr().Set(float(focal))


rp = rep.create.render_product("/World/cam", (W, H))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

# ------------------------------------------------------------------ drone camera and its synthetic data
DW, DH, D_FOCAL, D_APER = 960, 540, 28.0, 20.955
dcam = UsdGeom.Camera.Define(stage, "/World/drone_cam")
dcam.CreateFocalLengthAttr(D_FOCAL)
dcam.CreateHorizontalApertureAttr(D_APER)
dcam.CreateVerticalApertureAttr(D_APER * DH / DW)
dcam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 2000.0))
dcam_op = dcam.AddTransformOp()
drp = rep.create.render_product("/World/drone_cam", (DW, DH))
d_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
d_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
d_sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
for an in (d_rgb, d_depth, d_sem):
    an.attach([drp])


def updates(product, on):
    try:
        product.hydra_texture.set_updates_enabled(on)
    except Exception:
        pass


PALETTE = {"soil": (125, 88, 58), "lettuce": (70, 205, 80), "maize": (235, 205, 50), "weed": (235, 40, 45),
           "grass": (35, 115, 45), "robot": (45, 125, 245)}


def sem_image(d):
    ids = np.asarray(d["data"])
    out = np.full(ids.shape + (3,), 60, np.uint8)
    for k, v in d["info"].get("idToLabels", {}).items():
        lab = v.get("class", "") if isinstance(v, dict) else str(v)
        for name, col in PALETTE.items():
            if name in lab.lower():
                out[ids == int(k)] = col
    return out


def depth_image(d):
    z = np.asarray(d, np.float32)
    z = np.where(np.isfinite(z), z, 0)
    lo, hi = DRONE_Z_RANGE
    u = np.clip((z - lo) / (hi - lo), 0, 1)
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    xs = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(1 - u, xs, stops[:, c]) for c in range(3)], -1).astype(np.uint8)


def set_drone_cam(p, yaw):
    eye = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2] - 0.15))
    view = Gf.Matrix4d().SetLookAt(eye, eye - Gf.Vec3d(0, 0, 1), Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0))
    dcam_op.Set(view.GetInverse())
    return view


def project(view, p, focal, aper, w, h):
    c = view.Transform(Gf.Vec3d(*map(float, p)))
    if c[2] > -0.1:
        return None
    k = focal / aper * w
    return w / 2 + k * c[0] / -c[2], h / 2 - k * c[1] / -c[2], k / -c[2]


# ------------------------------------------------------------------ overlay
def font(size, bold=True):
    from PIL import ImageFont
    mpl = "/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/matplotlib/mpl-data/fonts/ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    cands = [f"/usr/share/fonts/truetype/dejavu/{name}", f"{mpl}/{name}"]
    for f in cands:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


F_TITLE, F_SUB, F_CAP, F_TAG = font(34), font(22, False), font(30), font(20)


def panel(draw, box, alpha=170):
    draw.rounded_rectangle(box, 10, fill=(18, 22, 20, alpha))


def overlay(img, caption, tags=(), legend=(), right=False):
    from PIL import ImageDraw
    im = Image.fromarray(img).convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    panel(d, (28, 26, 820, 112))
    d.rectangle((28, 26, 34, 112), fill=(120, 200, 70, 255))
    d.text((52, 34), "Farm digital twin  •  NVIDIA Isaac Sim", font=F_TITLE, fill=(255, 255, 255, 255))
    d.text((52, 76), "procedural crops • weeding rover • survey drone • synthetic data", font=F_SUB,
           fill=(215, 225, 215, 255))
    tw = d.textlength(caption, font=F_CAP)
    panel(d, (28, H - 104, 28 + tw + 48, H - 34))
    d.rectangle((28, H - 104, 34, H - 34), fill=(120, 200, 70, 255))
    d.text((52, H - 88), caption, font=F_CAP, fill=(255, 255, 255, 255))
    for (x, y, text) in tags:
        w_ = d.textlength(text, font=F_TAG)
        if right:                                   # x is the right edge of the panel the tag belongs to
            x = x - w_ - 50
        panel(d, (x, y, x + w_ + 20, y + 34), 190)
        d.text((x + 10, y + 5), text, font=F_TAG, fill=(255, 255, 255, 255))
    for (x, y, col, text) in legend:
        w_ = d.textlength(text, font=F_TAG)
        x = x - w_ - 84
        panel(d, (x, y, x + w_ + 54, y + 30), 190)
        d.rectangle((x + 10, y + 7, x + 26, y + 23), fill=tuple(col) + (255,))
        d.text((x + 36, y + 3), text, font=F_TAG, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))


def boxes(img, view):
    """Weed detections drawn from the scene's ground truth: every weed inside the drone's view gets a box."""
    from PIL import ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for p in WEED_POS:
        r = project(view, p, D_FOCAL, D_APER, DW, DH)
        if r is None or not (0 <= r[0] < DW and 0 <= r[1] < DH):
            continue
        s = 0.11 * r[2]
        d.rectangle((r[0] - s, r[1] - s, r[0] + s, r[1] + s), outline=(255, 50, 50), width=2)
    return np.asarray(im)


# ------------------------------------------------------------------ timeline
import timeline as tl                                  # noqa: E402

DRONE_Z_RANGE = (tl.DRONE_Z - 1.2, tl.DRONE_Z + 0.6)
# robots rest on their wheels: the highest ground under any wheel, plus how far the asset's wheels reach below its
# origin (Ridgeback 2.6 cm, Jackal 6.6 cm, from the assets' bounding boxes), so no wheel ever sinks into the soil
RIDGEBACK_WHEELS, RIDGEBACK_LIFT = [(sx * 0.32, sy * 0.28) for sx in (-1, 1) for sy in (-1, 1)], 0.026
JACKAL_WHEELS, JACKAL_LIFT = [(sx * 0.13, sy * 0.19) for sx in (-1, 1) for sy in (-1, 1)], 0.066


def rest_z(x, y, yaw, wheels, lift):
    c, s_ = math.cos(yaw), math.sin(yaw)
    return max(ground_z(x + c * wx - s_ * wy, y + s_ * wx + c * wy) for wx, wy in wheels) + lift


def rover_z(x, y):
    return rest_z(x, y, 0.0, RIDGEBACK_WHEELS, RIDGEBACK_LIFT)


ROVER_T = tl.Rover(rover_z, TARGETS, [w["home"][2] for w in WEEDS], ARM_MOUNT, BIN)
DRONE_T = tl.Drone()
SHOTS = tl.make_shots(ROVER_T, DRONE_T, ground_z)
PLAN = tl.plan(SHOTS)
print("frames", len(PLAN), "rover cycle", ROVER_T.cycle, flush=True)


def set_pose(W_, T):
    W_["t"].Set(Gf.Vec3d(*map(float, T[:3, 3])))
    sc = float(np.linalg.norm(T[:3, 0]))
    W_["s"].Set(Gf.Vec3f(sc, sc, sc))
    q = armk.mquat(T[:3, :3] / sc)
    W_["o"].Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))


def apply(f):
    si, t, u = PLAN[f]
    name, _, camf, caption, extra = SHOTS[si]
    x, q, a, k, tp = ROVER_T.at(t)
    ROVER["t"].Set(Gf.Vec3d(float(x), 0.0, rover_z(x, 0.0)))
    pose_arm(ROVER, q, a)
    spin_wheels(ROVER["wheels"], x - ROVER_T.stops[0] + 1.6)
    for i, Wd in enumerate(WEEDS):
        P = ROVER_T.weed_pose(i, t)
        set_pose(Wd, P if P is not None else tl.tz(*Wd["home"]))
    p, yaw = DRONE_T.at(t)
    DRONE["t"].Set(Gf.Vec3d(*map(float, p)))
    DRONE["r"].Set(math.degrees(yaw))
    for op, sgn in DRONE["spins"]:
        op.Set(float((sgn * PROP_STEP * f) % 360.0))
    jp, jyaw = tl.jackal_at(t, ground_z)
    jp[2] = rest_z(jp[0], jp[1], jyaw, JACKAL_WHEELS, JACKAL_LIFT)
    JACKAL["t"].Set(Gf.Vec3d(*map(float, jp)))
    JACKAL["r"].Set(math.degrees(jyaw))
    spin_wheels(JACKAL["wheels"], (jp[1] - tl.JACKAL_Y0) / max(math.sin(jyaw), 0.5))
    eye, tgt, focal = camf(t, u)
    set_camera(eye, tgt, focal)
    dview = set_drone_cam(p, yaw)
    return name, caption, extra, dview


def compose(name, caption, extra, dview):
    if extra.get("grid"):
        r_ = np.asarray(d_rgb.get_data())[..., :3]
        dep = depth_image(d_depth.get_data())
        sem = sem_image(d_sem.get_data())
        det = boxes(r_.copy(), dview)
        img = np.zeros((H, W, 3), np.uint8)
        img[:DH, :DW], img[:DH, DW:2 * DW] = r_, dep
        img[DH:2 * DH, :DW], img[DH:2 * DH, DW:2 * DW] = sem, det
        img[DH - 2:DH + 2] = 255
        img[:, DW - 2:DW + 2] = 255
        tags = [(DW, 130, "Drone camera: RGB"), (2 * DW, 130, "Depth"),
                (DW, DH + 20, "Semantic segmentation"), (2 * DW, DH + 20, "Weed detections (ground truth)")]
        legend = [(DW, DH + 66 + 34 * i, PALETTE[c], c) for i, c in enumerate(("lettuce", "weed", "soil", "robot"))]
        return overlay(img, caption, tags, legend, right=True)
    img = np.asarray(rgb.get_data())[..., :3].copy()
    tags = []
    if extra.get("pip"):
        pw, ph = 560, 315
        r_ = np.asarray(Image.fromarray(np.asarray(d_rgb.get_data())[..., :3]).resize((pw, ph)))
        sem = np.asarray(Image.fromarray(sem_image(d_sem.get_data())).resize((pw, ph), Image.NEAREST))
        x0 = W - pw - 30
        for j, (im_, label_) in enumerate(((r_, "Drone camera"), (sem, "Semantic labels"))):
            y0 = 30 + j * (ph + 14)
            img[y0 - 3:y0 + ph + 3, x0 - 3:x0 + pw + 3] = 255
            img[y0:y0 + ph, x0:x0 + pw] = im_
            tags.append((x0 + 10, y0 + 10, label_))
    return overlay(img, caption, tags)


os.makedirs(args.out, exist_ok=True)
if args.preview:
    frames = []
    for i in range(len(SHOTS)):
        idx = [f for f, pl in enumerate(PLAN) if pl[0] == i]
        frames.append(idx[len(idx) // 2])
elif args.frames:
    frames = [int(v) for v in args.frames.split(",")]
else:
    frames = list(range(args.start, len(PLAN) if args.end < 0 else min(args.end, len(PLAN))))

apply(frames[0])
for _ in range(6):                                     # warm-up: shaders, texture streaming
    rep.orchestrator.step(rt_subframes=24)
t0 = time.time()
drone_on = True
for n, f in enumerate(frames):
    name, caption, extra, dview = apply(f)
    want = bool(extra.get("pip") or extra.get("grid"))
    if want != drone_on:
        updates(drp, want)
        drone_on = want
    updates(rp, not extra.get("grid"))
    rep.orchestrator.step(rt_subframes=args.subframes)
    out = compose(name, caption, extra, dview)
    Image.fromarray(out).save(os.path.join(args.out, f"{f:04d}.png"))
    if n % 30 == 0 or args.preview:
        print(f"frame {f} ({n + 1}/{len(frames)}) {name} {(time.time() - t0) / (n + 1):.2f} s/frame", flush=True)
print("DONE", len(frames), "frames", flush=True)
app.close()

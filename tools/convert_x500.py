"""Convert PX4's Holybro X500 model (PX4-gazebo-models, BSD-3-Clause) into twin/assets/x500.npz.

The SDF places each mesh with a pose (x y z roll pitch yaw); this bakes those poses in, groups the frame's meshes
by material, and keeps each rotor (propeller + motor bell) in its own frame centred on the motor axis so the
render can spin it. Needs trimesh and pycollada:

  python tools/convert_x500.py <x500_base folder with model.sdf and meshes/> twin/assets/x500.npz
"""
import math
import os
import sys

import numpy as np
import trimesh

# material name in the .dae -> (rgb, roughness, metallic)
LOOK = {"CarbonFiber": ((0.045, 0.045, 0.05), 0.35, 0.0), "Metal": ((0.55, 0.56, 0.58), 0.3, 0.9),
        "LandingFoam": ((0.10, 0.10, 0.10), 0.8, 0.0), "LandingRubber": ((0.02, 0.02, 0.02), 0.7, 0.0),
        "RailsRubber": ((0.05, 0.05, 0.05), 0.7, 0.0), "FMUK66": ((0.85, 0.85, 0.86), 0.5, 0.0)}
MOTOR = ((0.10, 0.10, 0.11), 0.4, 0.6)
BELL = ((0.04, 0.04, 0.045), 0.35, 0.7)
PROP = ((0.03, 0.03, 0.03), 0.45, 0.0)


def pose(x, y, z, r, p, yw):
    """SDF pose -> 4x4 (fixed-axis roll, pitch, yaw: R = Rz Ry Rx)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(yw), math.sin(yw)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = (x, y, z)
    return T


def geoms(path):
    s = trimesh.load(path, force="scene")
    out = []
    for node in s.graph.nodes_geometry:
        T, gname = s.graph[node]
        g = s.geometry[gname].copy()
        g.apply_transform(T)
        out.append((gname.replace("-mesh", ""), g))
    return out


def baked(g, T, scale=1.0):
    v = np.asarray(g.vertices, float) * scale
    v = v @ T[:3, :3].T + T[:3, 3]
    return v.astype(np.float32), np.asarray(g.faces, np.int32)


def main(src, dst):
    m = os.path.join(src, "meshes")
    parts = []                                         # (name, rotor index or -1, verts, faces, rgb, rough, metal)
    for name, g in geoms(os.path.join(m, "NXP-HGD-CF.dae")):
        rgb, ro, me = LOOK.get(name, ((0.2, 0.2, 0.2), 0.5, 0.0))
        v, f = baked(g, pose(0, 0, 0.025, 0, 0, math.pi))
        parts.append((f"frame_{name}", -1, v, f, rgb, ro, me))
    motors = [(0.174, -0.174), (-0.174, 0.174), (0.174, 0.174), (-0.174, -0.174)]   # rotor_0..3
    spin = [+1, +1, -1, -1]                                                          # ccw, ccw, cw, cw
    for i, (x, y) in enumerate(motors):
        for _, g in geoms(os.path.join(m, "5010Base.dae")):
            v, f = baked(g, pose(x, y, 0.032, 0, 0, -0.45))
            parts.append((f"motor{i}", -1, v, f, *MOTOR))
        # rotor parts in the rotor's own frame (origin on the motor axis at the rotor link height)
        prop = "1345_prop_ccw.stl" if spin[i] > 0 else "1345_prop_cw.stl"
        g = trimesh.load(os.path.join(m, prop), force="mesh")
        v, f = baked(g, pose(-0.022, -0.14638461538461536, -0.016, 0, 0, 0), scale=0.8461538461538461)
        parts.append((f"prop{i}", i, v, f, *PROP))
        for _, g in geoms(os.path.join(m, "5010Bell.dae")):
            v, f = baked(g, pose(0, 0, -0.032, 0, 0, 0))
            parts.append((f"bell{i}", i, v, f, *BELL))
    data = {"names": np.array([p[0] for p in parts]), "rotor": np.array([p[1] for p in parts], np.int32),
            "rgb": np.array([p[4] for p in parts], np.float32), "rough": np.array([p[5] for p in parts], np.float32),
            "metal": np.array([p[6] for p in parts], np.float32),
            "rotor_pos": np.array([[x, y, 0.06] for x, y in motors], np.float32),
            "rotor_spin": np.array(spin, np.int32)}
    for k, p in enumerate(parts):
        data[f"v{k}"], data[f"f{k}"] = p[2], p[3]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, **data)
    allv = np.concatenate([p[2] for p in parts if p[1] < 0])
    print(f"{len(parts)} parts, {sum(len(p[3]) for p in parts)} triangles, frame bounds",
          np.round(allv.min(0), 3), np.round(allv.max(0), 3), "->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

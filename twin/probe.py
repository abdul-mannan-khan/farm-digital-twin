"""List the prim tree, sizes and xform ops of the robot assets used by farm.py (run once on the render box)."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

R = get_assets_root_path() + "/Isaac/Robots"
ASSETS = {"ridgeback": f"{R}/Clearpath/RidgebackUr/ridgeback_ur5.usd", "jackal": f"{R}/Clearpath/Jackal/jackal.usd",
          "quad": f"{R}/IsaacSim/Quadcopter/quadcopter.usd", "dingo": f"{R}/Clearpath/Dingo/dingo.usd"}
V = get_assets_root_path() + "/NVIDIA/Assets/Vegetation"
for n, f in (("beech", "Trees/American_Beech"), ("grassA", "Shrub/Grass_Short_A"), ("privet", "Shrub/Privet"),
             ("prairie", "Shrub/Prairie_Dropseed")):
    ASSETS[n] = f"{V}/{f}.usd"
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
for k, url in ASSETS.items():
    p = stage.DefinePrim(f"/World/{k}", "Xform")
    p.GetReferences().AddReference(url)
for _ in range(100):
    app.update()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for k in ASSETS:
    root = stage.GetPrimAtPath(f"/World/{k}")
    b = cache.ComputeWorldBound(root).ComputeAlignedRange()
    print(f"=== {k} bbox min {tuple(round(v, 3) for v in b.GetMin())} max {tuple(round(v, 3) for v in b.GetMax())}")
    vs = root.GetVariantSets()
    for n in vs.GetNames():
        print("  variant", n, vs.GetVariantSet(n).GetVariantNames(), "sel", vs.GetVariantSet(n).GetVariantSelection())
    for p in Usd.PrimRange(root):
        d = len(p.GetPath().pathString.split("/")) - 3
        if d > 3 or p.GetTypeName() in ("Shader", "Material", "GeomSubset"):
            continue
        ops = [o.GetOpName() for o in UsdGeom.Xformable(p).GetOrderedXformOps()] if p.IsA(UsdGeom.Xformable) else []
        bb = cache.ComputeWorldBound(p).ComputeAlignedRange()
        c = bb.GetMidpoint() if not bb.IsEmpty() else None
        print("  " * d, p.GetName(), p.GetTypeName(), ops,
              "" if c is None else f"c=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) s=({bb.GetSize()[0]:.3f},{bb.GetSize()[1]:.3f},{bb.GetSize()[2]:.3f})")
try:
    from isaacsim.core.utils.semantics import add_labels  # noqa: F401
    print("semantics: add_labels available")
except Exception as e:
    print("semantics: add_labels missing", e)
app.close()

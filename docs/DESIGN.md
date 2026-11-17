# Design notes

How the farm digital twin is built, step by step, with the numbers behind each choice.

## 1. Pipeline

```
plants.py  ──► leaf meshes + textures (numpy, PIL)
                     │
farm.py    ──► Isaac Sim stage: terrain ─ crops (point instancers) ─ margins ─ sky ─ robots
                     │
timeline.py ─► per-frame state: rover x + arm joints + weed poses, drone pose, Jackal pose, camera shot
                     │
farm.py    ──► Replicator: main camera (1920x1080) + drone camera (960x540: RGB, depth, semantics)
                     │                 ↓
                     └─────────► PIL compositing: picture-in-picture, 2x2 data grid, titles, captions
                                       ↓
run_all.sh / entrypoint.sh ─► ffmpeg: 1080p mp4 (crf 18), web mp4 (crf 26), GIF (2x speed, 12 fps)
```

Nothing is simulated with physics: every pose is computed, so a frame can be rendered on its own and a render can be
split into chunks (`--start/--end`) or single frames (`--frames`) and stays reproducible (`--seed`).

## 2. Field layout (metres; x east, y north, z up)

| Block | Extent | Structure |
|---|---|---|
| Lettuce | x -44..-2, y -28..28 | raised beds along x: 1.6 m pitch, 1.0 m flat top, 14 cm high, 18 cm shoulders; 3 rows per bed at 30 cm, plants every 30 cm (middle row offset 15 cm); every third bed red lettuce |
| Tramline | y -0.6..0.6 in the lettuce | flat, where the weeding rover drives |
| Farm track | x -1.2..2.0 | compacted, two wheel ruts 4 cm deep |
| Maize | x 3..44, y -28..28 | ridges at 76 cm, plants every 18 cm |
| Headland | 1-4 m round the field | grass texture plus ~9,000 instanced grass clumps |
| Hedgerows | north, south, west | NVIDIA trees (green species only) and shrubs every ~5.5 m |
| Meadow | to 700 m | follows the field undulation 8 cm lower, rising into low hills beyond 150 m |

Final counts: 13,658 lettuce, 15,975 maize, 1,757 weeds (plus 12 weeds the rover pulls), about 3% planting gaps.

The terrain is a single mesh at 0.4 m along the beds and 8 cm across them (enough to resolve the bed shoulders),
with per-face material subsets (soil, track, headland) whose boundaries lie on grid lines so the material edges are
straight. Smooth vertex normals are written for every mesh; without them RTX shades the leaves and beds faceted.

## 3. Procedural plants (`twin/plants.py`)

All plant organs are swept surfaces: a spine curve from base to tip, a frame across the blade, and an offset field
across the blade (`sweep_leaf`). Different organs only change the spine, width profile and offsets.

* **Lettuce leaf**: spine pitched up at 84-34 degrees (inner to outer) and bent by -70..+70 degrees along its length,
  so inner leaves curl over the heart and outer leaves arch out and lie down; obovate width profile (narrow petiole,
  widest at ~70%); cup across the blade (edges raised ~ v^2); a ruffle at the edge (~ |v|^4 sin(...)). 16-24 leaves
  placed at the golden angle (137.5 deg). 16 x 13 vertices per leaf.
* **Maize**: tapered stalk tube, 11-14 strap leaves alternating 180 degrees (two ranks in one plane, oriented across
  the row), each leaving the stalk at 45-70 degrees and drooping to -35..+25 degrees at the tip with a V-fold midrib,
  slight twist and a wavy edge; a tassel of 10-16 drooping branches; 1-2 ears in husks with silk.
* **Weed**: 7-12 narrow toothed leaves lying almost flat.

Textures (`leaf_texture`) are painted per organ: colour gradients from base to tip, a midrib, lateral veins at an
angle, low-frequency noise; senescent tips on the lowest maize leaves; anthocyanin on red lettuce edges. Several
variants per species (5 green lettuce, 3 red, 6 maize, 3 weed) are instanced with random yaw and 85-115% scale.

## 4. Materials and light

* OmniPBR materials with Poly Haven photo-scanned maps (colour, OpenGL normal, roughness; normal and roughness maps
  read as raw data). With a texture set OmniPBR ignores the colour constant, so tints go through `diffuse_tint`.
* Dome light with the Poly Haven `farm_field` HDRI (4k EXR) at intensity 1000 and a distant sun (3000, 0.6 degree
  disc) for crisp shadows; RTX real-time renderer with indirect diffuse, ambient occlusion, reflections, DLSS and
  auto exposure; 8 subframes per frame.

## 5. Robots

| Robot | Asset | Placement |
|---|---|---|
| Weeding rover | Clearpath Ridgeback (its UR5 deactivated) + UR5e with Robotiq 2F-85 on a 10 cm pedestal at (0.12, 0.34, 0.38) on the deck, weed bin, camera mast | tramline, y = 0 |
| Survey drone | quadrotor (Crazyflie model scaled to 0.65 m) | 11 m above the field |
| Scout | Clearpath Jackal | farm track |

The assets carry rigid bodies, joints and articulation roots for physics; `strip_physics` disables them so the
robots follow the timeline. Robots rest on their wheels: the body height is the highest ground under any of the four
wheels plus the depth the asset's wheels reach below its origin (Ridgeback 2.6 cm, Jackal 6.6 cm), so no wheel
sinks into the soil on the bed shoulders or in the track ruts.

## 6. Weeding cycle (`twin/timeline.py`)

The rover stops so the arm base is level with the next weed (weeds grow in gaps of the first lettuce row, 0.65 m
to the side and ~0.21 m below the arm base). One cycle (6.0 s) plus a 2.2 s drive:

| Move | s | Path |
|---|---|---|
| to_src | 1.00 | arc round the base from home to 22 cm above the weed |
| down | 0.55 | straight down |
| close | 0.35 | fingers close on the weed (Robotiq linkage from `arm.fingers`) |
| up | 0.55 | straight up with the weed |
| to_dst | 1.35 | arc round the base, small hop over the deck, to above the bin |
| down_dst, open, up_dst | 1.00 | lower into the bin, release, rise |
| home | 1.20 | arc back |

Each move is sampled at 20 points along its Cartesian path; IK (damped least squares, log-map orientation error) is
solved at each point seeded with the previous solution and unwrapped to the nearest 2 pi, so the whole cycle stays
on one posture branch from a fixed elbow-up home. The jaw yaw follows the arm's azimuth so the last wrist joint
barely moves. Checked offline: largest joint step 7.8 degrees per frame, largest tool step 6.6 cm per frame.

The weed's pose relative to the tool at the grasp is stored, so while held it hangs from the gripper exactly as it
was picked; on release it drops into the bin over 0.25 s, shrinking to 60% (wilted), at a slot that fills the bin
in a 3 x 2 grid, then rides with the rover.

## 7. Drone and synthetic data

The drone flies a lawnmower survey (8 legs, 48 m long, 8 m apart) at 6 m/s with corners rounded by Chaikin
subdivision; heading follows the path. Its camera looks straight down (horizontal FOV ~41 degrees, ~8 m footprint).

| Panel | Source |
|---|---|
| RGB | Replicator `rgb` annotator |
| Depth | `distance_to_image_plane`, coloured with a viridis-like ramp over 9.8-11.6 m |
| Semantic segmentation | `semantic_segmentation`; classes from labels on the plant prototypes (lettuce, maize, weed), the terrain (soil), the meadow (grass) and the robots |
| Weed detections | every weed's ground-truth position projected through the same pinhole model; box size from the projected 11 cm weed radius |

The drone render product is switched off outside the shots that show its data, and the main one during the 2x2 grid.

## 8. Shots (30 fps, 1800 frames)

| Frames | Shot | Camera |
|---|---|---|
| 0-209 | establish | crane down from 95 m SW of the field |
| 210-539 | chase | behind and beside the drone, picture-in-picture of its RGB and labels |
| 540-839 | rover_wide | orbit round the rover |
| 840-1109 | rover_close | over the gripper |
| 1110-1379 | data | 2x2 drone data grid |
| 1380-1589 | maize | beside the Jackal, maize wall behind |
| 1590-1799 | closing | rising pull-back over the field |

## 9. Performance

On an RTX 4090 (Isaac Sim 6.2 container): scene build ~30 s, first frames ~2 min (shader compilation), then
~0.95 s per 1080p frame with 8 subframes, so ~30 min for the film plus ~3 min of encoding.

## 10. Known limitations

* Plants are static (no wind) and the rover's arm does not avoid the lettuce leaves explicitly; the paths stay
  above them by construction.
* The weeding rover does not pitch or roll with the terrain (the tramline is nearly flat).
* Weed boxes are ground truth, not a trained detector; the point of the panel is the data a detector would be
  trained on.

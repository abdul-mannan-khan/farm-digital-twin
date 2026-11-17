"""Checks that run without Isaac Sim: plant meshes, arm reach and smoothness of the weeding cycle, drone path."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "twin"))
import arm  # noqa: E402
import plants  # noqa: E402
import timeline as tl  # noqa: E402

MOUNT, BIN = (0.12, 0.34, 0.38), (-0.24, -0.02, 0.28)


def ground(x, y):
    return 0.05 * math.sin(x / 9) + (0.14 if y > 0.7 else 0.0)


def rover():
    xs = (-36.1, -34.6, -33.2, -31.6, -30.1, -28.5, -27.0, -25.4)
    targets = np.array([[x, 0.99 + 0.015 * math.sin(3.1 * x)] for x in xs])
    return tl.Rover(ground, targets, [ground(x, y) for x, y in targets], MOUNT, BIN)


def test_plants_are_valid_meshes():
    rng = np.random.default_rng(0)
    for make in (plants.lettuce, plants.maize, plants.weed):
        for p in make(rng):
            assert p["faces"].max() < len(p["points"])
            assert np.isfinite(p["points"]).all()
            assert p["uv"].shape == (len(p["points"]), 2)
            assert p["tex"] in plants.TEXTURES


def test_plant_sizes_are_botanical():
    rng = np.random.default_rng(1)
    let = np.concatenate([p["points"] for p in plants.lettuce(rng)])
    assert 0.15 < np.ptp(let[:, 0]) < 0.45 and 0.08 < let[:, 2].max() < 0.25      # head lettuce
    mai = np.concatenate([p["points"] for p in plants.maize(rng, height=2.0)])
    assert 1.8 < mai[:, 2].max() < 2.6                                              # maize with tassel


def test_weeding_cycle_is_smooth():
    R = rover()
    prev_q, prev_p = None, None
    for f in range(int(R.cycle * len(R.pp) * tl.FPS)):
        x, q, a, k, tp = R.at(f / tl.FPS)
        p = arm.fk(q)["tcp"][:3, 3]
        if prev_q is not None:
            dq = np.abs((q - prev_q + math.pi) % (2 * math.pi) - math.pi).max()
            assert math.degrees(dq) < 10.0, f"joint jump at frame {f}"
            assert np.linalg.norm(p - prev_p) < 0.08, f"tool jump at frame {f}"
        prev_q, prev_p = q, p


def test_weed_follows_gripper_and_lands_in_bin():
    R = rover()
    t_grasp = R.cycle + tl.DRIVE_T + 2.2               # weed 1 is lifted by then
    held = R.weed_pose(1, t_grasp)
    x, q, *_ = R.at(t_grasp)
    tcp = R.world(x) @ tl.tz(*MOUNT) @ arm.fk(q)["tcp"]
    assert held is not None and np.linalg.norm(held[:3, 3] - tcp[:3, 3]) < 0.08
    later = R.weed_pose(1, 3 * R.cycle)
    x, *_ = R.at(3 * R.cycle)
    in_bin = np.linalg.inv(R.world(x)) @ later
    assert abs(in_bin[0, 3] - BIN[0]) < 0.15 and abs(in_bin[1, 3] - BIN[1]) < 0.12
    assert R.weed_pose(5, 0.0) is None                  # weeds not reached yet stay where they grew


def test_drone_survey_constant_speed():
    D = tl.Drone()
    p = np.array([D.at(t)[0][:2] for t in np.arange(0, 40, 0.5)])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.5
    assert (v < tl.DRONE_SPEED * 1.02).all()                  # never faster than cruise (chords cut the turns)
    assert abs(np.median(v) - tl.DRONE_SPEED) < 0.05 * tl.DRONE_SPEED


def test_film_is_one_minute():
    R = rover()
    shots = tl.make_shots(R, tl.Drone(), ground)
    assert len(tl.plan(shots)) == 60 * tl.FPS

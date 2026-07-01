#!/usr/bin/env python3
"""Create safer Isaac Sim USD scene copies for grasp execution tests.

The script never edits the source USD files. It copies selected scenes into an
output directory, then edits only the copies.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except Exception as exc:  # pragma: no cover - this depends on Isaac Sim env.
    raise RuntimeError(
        "Could not import pxr. Run this script with Isaac Sim's python/USD libs. "
        "See tools/run_create_simsafe_usd_copies.sh."
    ) from exc


DEFAULT_SCENE_DIR = Path("/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib")
DEFAULT_OUTPUT_DIR = Path("/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib_simsafe")

SCENE_PATTERNS = {
    "knife": ("cam{cam_id}_r.usd",),
    "hammer": ("hammer_cam{cam_id}.usd",),
    "brush": ("brush_cam{cam_id}.usd",),
    "spoon": ("spoon_cam{cam_id}.usd",),
    "drill": ("drill_cam{cam_id}.usd",),
    "fork": ("fork_cam{cam_id}.usd",),
}

TARGET_KEYWORDS = {
    "knife": ("knife",),
    "hammer": ("hammer",),
    "brush": ("brush",),
    "spoon": ("spoon",),
    "drill": ("drill",),
    "fork": ("fork",),
}

TARGET_EXCLUDE_KEYWORDS = {
    "knife": ("holder", "holders", "kitchen"),
}

ESSENTIAL_TOP_KEYWORDS = (
    "ground",
    "camera",
    "light",
    "looks",
    "render",
    "physics",
    "robot",
    "franka",
    "panda",
    "so101",
)

SUPPORT_KEYWORDS = (
    "wood",
    "block",
    "stick",
    "holder",
    "support",
    "stand",
)


@dataclass
class SceneEditReport:
    source: str
    output: str
    object_name: str
    target_paths: list[str]
    disabled_collision_prims: int
    disabled_rigid_body_prims: int
    scaled_support_prims: list[str]
    target_z_offset_m: float
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy USD scenes and make safer simulation variants without touching originals."
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["knife", "hammer", "brush", "spoon", "drill"],
        choices=sorted(SCENE_PATTERNS.keys()),
    )
    parser.add_argument("--cam-ids", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument(
        "--target-z-offset",
        type=float,
        default=0.0,
        help="Add this local Z translation to target top-level prims in copied scenes.",
    )
    parser.add_argument(
        "--disable-clutter-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable collision and rigid body on non-target, non-essential scene props.",
    )
    parser.add_argument(
        "--disable-support-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also disable collision on support-like props such as wood blocks.",
    )
    parser.add_argument(
        "--support-scale",
        type=float,
        default=1.0,
        help="Uniformly scale support-like top-level props in the copied scene. 1.0 leaves them unchanged.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite existing copied USD files in output-dir.",
    )
    return parser.parse_args()


def lower_name(prim: Usd.Prim) -> str:
    return prim.GetName().lower()


def name_has_any(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def is_target_top_prim(prim: Usd.Prim, object_name: str) -> bool:
    name = lower_name(prim)
    if name_has_any(name, TARGET_EXCLUDE_KEYWORDS.get(object_name, ())):
        return False
    return name_has_any(name, TARGET_KEYWORDS[object_name])


def is_essential_top_prim(prim: Usd.Prim) -> bool:
    return name_has_any(lower_name(prim), ESSENTIAL_TOP_KEYWORDS)


def is_support_top_prim(prim: Usd.Prim) -> bool:
    return name_has_any(lower_name(prim), SUPPORT_KEYWORDS)


def iter_world_children(stage: Usd.Stage) -> list[Usd.Prim]:
    world = stage.GetPrimAtPath("/World")
    if not world or not world.IsValid():
        return []
    return [child for child in world.GetChildren() if child.IsValid()]


def has_collision_api(prim: Usd.Prim) -> bool:
    try:
        return prim.HasAPI(UsdPhysics.CollisionAPI)
    except Exception:
        return False


def has_rigid_body_api(prim: Usd.Prim) -> bool:
    try:
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)
    except Exception:
        return False


def disable_collision_recursively(root: Usd.Prim) -> tuple[int, int]:
    collision_count = 0
    rigid_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or prim.IsPseudoRoot():
            continue

        should_apply_collision = has_collision_api(prim) or prim.IsA(UsdGeom.Gprim)
        if should_apply_collision:
            try:
                collision_api = (
                    UsdPhysics.CollisionAPI(prim)
                    if has_collision_api(prim)
                    else UsdPhysics.CollisionAPI.Apply(prim)
                )
                collision_api.CreateCollisionEnabledAttr(False)
                collision_count += 1
            except Exception:
                pass

        if has_rigid_body_api(prim):
            try:
                rigid_api = UsdPhysics.RigidBodyAPI(prim)
                rigid_api.CreateRigidBodyEnabledAttr(False)
                rigid_count += 1
            except Exception:
                pass
    return collision_count, rigid_count


def add_local_z_offset(prim: Usd.Prim, z_offset: float) -> bool:
    if abs(z_offset) < 1e-12:
        return False

    xform = UsdGeom.Xformable(prim)
    if not xform:
        attr_name = "xformOp:translate:simsafe"
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Double3)
        value = attr.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
        attr.Set(Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]) + z_offset))
        order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
        order = list(order_attr.Get() or [])
        if attr_name not in order:
            order.append(attr_name)
            order_attr.Set(order)
        return True

    translate_ops = [
        op for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if translate_ops:
        op = translate_ops[-1]
    else:
        op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)

    value = op.Get()
    if value is None:
        value = Gf.Vec3d(0.0, 0.0, 0.0)
    op.Set(Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]) + z_offset))
    return True


def multiply_support_scale(prim: Usd.Prim, scale: float) -> bool:
    if abs(scale - 1.0) < 1e-12:
        return False
    if scale <= 0:
        raise ValueError("--support-scale must be positive")

    xform = UsdGeom.Xformable(prim)
    if not xform:
        attr_name = "xformOp:scale:simsafe"
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float3)
        value = attr.Get() or Gf.Vec3f(1.0, 1.0, 1.0)
        attr.Set(Gf.Vec3f(float(value[0]) * scale, float(value[1]) * scale, float(value[2]) * scale))
        order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
        order = list(order_attr.Get() or [])
        if attr_name not in order:
            order.append(attr_name)
            order_attr.Set(order)
        return True

    scale_ops = [
        op for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if scale_ops:
        op = scale_ops[-1]
        value = op.Get()
        if value is None:
            value = Gf.Vec3f(1.0, 1.0, 1.0)
        op.Set(Gf.Vec3f(float(value[0]) * scale, float(value[1]) * scale, float(value[2]) * scale))
    else:
        op = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
        op.Set(Gf.Vec3f(scale, scale, scale))
    return True


def edit_scene_copy(
    usd_path: Path,
    object_name: str,
    args: argparse.Namespace,
) -> SceneEditReport:
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Could not open copied USD: {usd_path}")

    warnings: list[str] = []
    world_children = iter_world_children(stage)
    if not world_children:
        warnings.append("No /World children found.")

    target_prims = [prim for prim in world_children if is_target_top_prim(prim, object_name)]
    if not target_prims:
        warnings.append(f"No target top prim matched object '{object_name}'.")

    for prim in target_prims:
        add_local_z_offset(prim, args.target_z_offset)

    disabled_collision = 0
    disabled_rigid = 0
    scaled_supports: list[str] = []
    if args.disable_clutter_collision:
        for prim in world_children:
            if is_target_top_prim(prim, object_name) or is_essential_top_prim(prim):
                continue

            support_like = is_support_top_prim(prim)
            if support_like and abs(args.support_scale - 1.0) > 1e-12:
                if multiply_support_scale(prim, args.support_scale):
                    scaled_supports.append(str(prim.GetPath()))

            if support_like and not args.disable_support_collision:
                continue

            c_count, r_count = disable_collision_recursively(prim)
            disabled_collision += c_count
            disabled_rigid += r_count

    stage.GetRootLayer().Save()

    return SceneEditReport(
        source="",
        output=str(usd_path),
        object_name=object_name,
        target_paths=[str(prim.GetPath()) for prim in target_prims],
        disabled_collision_prims=disabled_collision,
        disabled_rigid_body_prims=disabled_rigid,
        scaled_support_prims=scaled_supports,
        target_z_offset_m=args.target_z_offset,
        warnings=warnings,
    )


def scene_files_for_object(scene_dir: Path, object_name: str, cam_ids: list[int]) -> list[Path]:
    files: list[Path] = []
    for cam_id in cam_ids:
        for pattern in SCENE_PATTERNS[object_name]:
            path = scene_dir / pattern.format(cam_id=cam_id)
            if path.exists():
                files.append(path)
    return files


def main() -> None:
    args = parse_args()
    args.scene_dir = args.scene_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser()

    if not args.scene_dir.exists():
        raise FileNotFoundError(f"Scene dir not found: {args.scene_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[SceneEditReport] = []
    missing: dict[str, list[str]] = {}

    for object_name in args.objects:
        sources = scene_files_for_object(args.scene_dir, object_name, args.cam_ids)
        found_names = {source.name for source in sources}
        expected_names = [
            pattern.format(cam_id=cam_id)
            for cam_id in args.cam_ids
            for pattern in SCENE_PATTERNS[object_name]
        ]
        missing[object_name] = [name for name in expected_names if name not in found_names]

        for source in sources:
            dest = args.output_dir / source.name
            if dest.exists() and not args.overwrite_output:
                raise FileExistsError(
                    f"Output exists: {dest}. Use --overwrite-output or choose another --output-dir."
                )

            shutil.copy2(source, dest)
            report = edit_scene_copy(dest, object_name, args)
            report.source = str(source)
            reports.append(report)
            print(
                f"OK {source.name}: target={report.target_paths or 'NONE'} "
                f"disabled_collision={report.disabled_collision_prims} "
                f"disabled_rigid={report.disabled_rigid_body_prims} "
                f"scaled_supports={len(report.scaled_support_prims)}"
            )

    report_path = args.output_dir / "simsafe_edit_report.json"
    payload = {
        "scene_dir": str(args.scene_dir),
        "output_dir": str(args.output_dir),
        "objects": args.objects,
        "cam_ids": args.cam_ids,
        "target_z_offset": args.target_z_offset,
        "disable_clutter_collision": args.disable_clutter_collision,
        "disable_support_collision": args.disable_support_collision,
        "support_scale": args.support_scale,
        "missing": missing,
        "reports": [report.__dict__ for report in reports],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 70)
    print(f"Wrote {len(reports)} copied USD scenes to: {args.output_dir}")
    print(f"Report: {report_path}")
    for object_name, names in missing.items():
        if names:
            preview = ", ".join(names[:5])
            suffix = " ..." if len(names) > 5 else ""
            print(f"Missing {object_name}: {preview}{suffix}")


if __name__ == "__main__":
    main()

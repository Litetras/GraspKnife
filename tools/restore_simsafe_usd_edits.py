#!/usr/bin/env python3
"""Remove simsafe USD edits that were authored by create_simsafe_usd_copies.py.

This is a repair tool. By default it runs in dry-run mode and only prints what
would be changed. Pass --apply to save changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from pxr import Sdf, Usd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import pxr. Run this with Isaac Sim USD libs, e.g. via "
        "tools/run_restore_simsafe_usd_edits.sh."
    ) from exc


DEFAULT_SCENE_DIR = Path("/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib")
SIMSAFE_XFORM_OPS = {
    "xformOp:translate:simsafe",
    "xformOp:scale:simsafe",
}
SIMSAFE_CREATED_ATTRS = SIMSAFE_XFORM_OPS | {
    "physics:collisionEnabled",
    "physics:rigidBodyEnabled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore USD files by removing simsafe-authored opinions.")
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--files", nargs="*", default=None, help="Specific USD filenames to inspect.")
    parser.add_argument("--apply", action="store_true", help="Actually save changes. Default is dry-run.")
    parser.add_argument(
        "--clear-physics-disabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clear authored False values on physics:collisionEnabled and "
            "physics:rigidBodyEnabled. This is the inverse of the simsafe script."
        ),
    )
    return parser.parse_args()


def remove_from_xform_order(prim: Usd.Prim, op_names: set[str]) -> bool:
    attr = prim.GetAttribute("xformOpOrder")
    if not attr:
        return False
    value = attr.Get()
    if value is None:
        return False
    new_value = [token for token in value if str(token) not in op_names]
    if list(value) == new_value:
        return False
    attr.Set(new_value)
    return True


def attr_authored_false(attr: Usd.Attribute) -> bool:
    if not attr:
        return False
    if not attr.HasAuthoredValueOpinion():
        return False
    value = attr.Get()
    return value is False


def restore_stage(path: Path, apply: bool, clear_physics_disabled: bool) -> dict:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Could not open USD: {path}")

    removed_props: list[str] = []
    cleaned_orders: list[str] = []
    cleared_false_attrs: list[str] = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        for prop_name in SIMSAFE_XFORM_OPS:
            prop = prim.GetProperty(prop_name)
            if prop:
                removed_props.append(f"{prim.GetPath()}.{prop_name}")
                if apply:
                    prim.RemoveProperty(prop_name)

        if remove_from_xform_order(prim, SIMSAFE_XFORM_OPS):
            cleaned_orders.append(str(prim.GetPath()))

        if clear_physics_disabled:
            for attr_name in ("physics:collisionEnabled", "physics:rigidBodyEnabled"):
                attr = prim.GetAttribute(attr_name)
                if attr_authored_false(attr):
                    cleared_false_attrs.append(f"{prim.GetPath()}.{attr_name}")
                    if apply:
                        attr.Clear()

    if apply and (removed_props or cleaned_orders or cleared_false_attrs):
        stage.GetRootLayer().Save()

    return {
        "file": str(path),
        "removed_simsafe_xform_props": removed_props,
        "cleaned_xform_orders": cleaned_orders,
        "cleared_authored_false_physics_attrs": cleared_false_attrs,
        "would_save": bool(removed_props or cleaned_orders or cleared_false_attrs),
    }


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.exists():
        raise FileNotFoundError(scene_dir)

    if args.files:
        usd_files = [scene_dir / name for name in args.files]
    else:
        usd_files = sorted(scene_dir.glob("*.usd"))

    reports = []
    for path in usd_files:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        report = restore_stage(path, apply=args.apply, clear_physics_disabled=args.clear_physics_disabled)
        reports.append(report)
        if report["would_save"]:
            mode = "RESTORED" if args.apply else "WOULD_RESTORE"
            print(
                f"{mode} {path.name}: "
                f"simsafe_props={len(report['removed_simsafe_xform_props'])}, "
                f"xform_orders={len(report['cleaned_xform_orders'])}, "
                f"physics_false={len(report['cleared_authored_false_physics_attrs'])}"
            )

    summary = {
        "scene_dir": str(scene_dir),
        "apply": args.apply,
        "files_checked": len(usd_files),
        "files_with_changes": sum(1 for item in reports if item["would_save"]),
        "reports": reports,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

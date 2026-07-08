#!/usr/bin/env python3
"""Summarize IsaacSim LODGrasp/baseline trial_results.csv files.

This script is intentionally independent from IsaacSim. It only reads CSV files.

Typical usage:
  python tools/summarize_sim_eval_results.py \
    --run-name overnight_fork_spoon_drill_20260705_003520 \
    --output-csv /tmp/sim_summary.csv

Metrics:
  generated_rate:
    A valid 6-DoF grasp pose was produced. This is useful when physics/lift is
    unreliable or disabled.

  physics_success_rate:
    Directly uses success/physics_success from trial_results.csv.

  manual_pose_rate:
    Uses pose_correct_manual if you have filled manual review columns.
    Denominator is reviewed rows only.

  manual_position_rate / manual_direction_rate:
    Uses position_correct_manual / direction_correct_manual if present.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRUE_VALUES = {"1", "true", "yes", "y", "ok", "success", "succeed"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed"}


@dataclass
class Trial:
    method: str
    object_name: str
    task: str
    scene_id: str
    trial_id: str
    row: dict
    csv_path: Path


def parse_bool(value: object):
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def has_generated_grasp(row: dict) -> bool:
    return bool(
        str(row.get("best_pose_exec_world") or "").strip()
        or str(row.get("best_pose_raw_camera") or "").strip()
    )


def is_no_grasp_failure(row: dict) -> bool:
    if has_generated_grasp(row):
        return False
    reason = str(row.get("fail_reason") or "").lower()
    return (
        "inference_failed" in reason
        or "无有效抓取" in reason
        or "未生成抓取" in reason
        or "no valid" in reason
        or "no grasp" in reason
    )


def discover_run_roots(run_name: str, search_base: Path) -> list[Path]:
    pattern = f"**/batch_test_results_refactored/{run_name}"
    roots = sorted(p for p in search_base.glob(pattern) if p.is_dir())
    # Prefer roots that actually contain CSVs.
    return [p for p in roots if any(p.glob("*/*/*/trial_results.csv"))]


def iter_trials(root: Path) -> Iterable[Trial]:
    for csv_path in sorted(root.glob("*/*/*/trial_results.csv")):
        rel = csv_path.relative_to(root)
        if len(rel.parts) < 4:
            continue
        method, object_name, task = rel.parts[0], rel.parts[1], rel.parts[2]
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Trial(
                    method=method,
                    object_name=row.get("object") or object_name,
                    task=row.get("task") or task,
                    scene_id=row.get("scene_id") or "",
                    trial_id=row.get("trial_id") or "",
                    row=row,
                    csv_path=csv_path,
                )


def pct(num: int, den: int) -> str:
    if den <= 0:
        return "NA"
    return f"{num / den * 100:.1f}%"


def summarize_group(trials: list[Trial], ignore_no_grasp: bool = False) -> dict:
    denom_trials = [
        t for t in trials
        if not (ignore_no_grasp and is_no_grasp_failure(t.row))
    ]
    den = len(denom_trials)

    generated = sum(has_generated_grasp(t.row) for t in denom_trials)
    physics = sum(parse_bool(t.row.get("success")) is True for t in denom_trials)
    physics2 = sum(parse_bool(t.row.get("physics_success")) is True for t in denom_trials)

    manual_pose_vals = [parse_bool(t.row.get("pose_correct_manual")) for t in denom_trials]
    manual_pos_vals = [parse_bool(t.row.get("position_correct_manual")) for t in denom_trials]
    manual_dir_vals = [parse_bool(t.row.get("direction_correct_manual")) for t in denom_trials]

    manual_pose_reviewed = [v for v in manual_pose_vals if v is not None]
    manual_pos_reviewed = [v for v in manual_pos_vals if v is not None]
    manual_dir_reviewed = [v for v in manual_dir_vals if v is not None]

    fail_reasons = Counter()
    for t in denom_trials:
        if parse_bool(t.row.get("success")) is True:
            fail_reasons["ok"] += 1
        else:
            fail_reasons[t.row.get("fail_reason") or "unknown"] += 1

    return {
        "n": len(trials),
        "den": den,
        "ignored_no_grasp": len(trials) - den,
        "generated": generated,
        "physics_success": physics,
        "physics_success_alt": physics2,
        "manual_pose_reviewed": len(manual_pose_reviewed),
        "manual_pose_success": sum(manual_pose_reviewed),
        "manual_position_reviewed": len(manual_pos_reviewed),
        "manual_position_success": sum(manual_pos_reviewed),
        "manual_direction_reviewed": len(manual_dir_reviewed),
        "manual_direction_success": sum(manual_dir_reviewed),
        "fail_reasons": fail_reasons,
    }


def print_summary(title: str, groups: dict[tuple, list[Trial]], ignore_no_grasp: bool):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    for key in sorted(groups):
        stats = summarize_group(groups[key], ignore_no_grasp=ignore_no_grasp)
        reason_text = "; ".join(
            f"{k}:{v}" for k, v in stats["fail_reasons"].most_common(3)
        )
        key_text = " / ".join(str(x) for x in key)
        print(
            f"{key_text:<42} "
            f"n={stats['n']:3d} den={stats['den']:3d} "
            f"generated={pct(stats['generated'], stats['den']):>6} "
            f"physics={pct(stats['physics_success'], stats['den']):>6} "
            f"manual_pose={pct(stats['manual_pose_success'], stats['manual_pose_reviewed']):>6} "
            f"manual_pos={pct(stats['manual_position_success'], stats['manual_position_reviewed']):>6} "
            f"manual_dir={pct(stats['manual_direction_success'], stats['manual_direction_reviewed']):>6} "
            f"ignored_no_grasp={stats['ignored_no_grasp']:3d} "
            f"fail_top={reason_text}"
        )


def write_csv(path: Path, groups: dict[tuple, list[Trial]], ignore_no_grasp: bool):
    fieldnames = [
        "group",
        "method",
        "object",
        "task",
        "n",
        "den",
        "ignored_no_grasp",
        "generated_success",
        "generated_rate",
        "physics_success",
        "physics_rate",
        "manual_pose_reviewed",
        "manual_pose_success",
        "manual_pose_rate",
        "manual_position_reviewed",
        "manual_position_success",
        "manual_position_rate",
        "manual_direction_reviewed",
        "manual_direction_success",
        "manual_direction_rate",
        "fail_reason_top3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            stats = summarize_group(groups[key], ignore_no_grasp=ignore_no_grasp)
            method = key[0] if len(key) > 0 else ""
            obj = key[1] if len(key) > 1 else ""
            task = key[2] if len(key) > 2 else ""
            writer.writerow({
                "group": " / ".join(str(x) for x in key),
                "method": method,
                "object": obj,
                "task": task,
                "n": stats["n"],
                "den": stats["den"],
                "ignored_no_grasp": stats["ignored_no_grasp"],
                "generated_success": stats["generated"],
                "generated_rate": pct(stats["generated"], stats["den"]),
                "physics_success": stats["physics_success"],
                "physics_rate": pct(stats["physics_success"], stats["den"]),
                "manual_pose_reviewed": stats["manual_pose_reviewed"],
                "manual_pose_success": stats["manual_pose_success"],
                "manual_pose_rate": pct(stats["manual_pose_success"], stats["manual_pose_reviewed"]),
                "manual_position_reviewed": stats["manual_position_reviewed"],
                "manual_position_success": stats["manual_position_success"],
                "manual_position_rate": pct(stats["manual_position_success"], stats["manual_position_reviewed"]),
                "manual_direction_reviewed": stats["manual_direction_reviewed"],
                "manual_direction_success": stats["manual_direction_success"],
                "manual_direction_rate": pct(stats["manual_direction_success"], stats["manual_direction_reviewed"]),
                "fail_reason_top3": "; ".join(
                    f"{k}:{v}" for k, v in stats["fail_reasons"].most_common(3)
                ),
            })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--run-name", default="")
    parser.add_argument("--search-base", type=Path, default=Path("/home/zyp/IsaacLab"))
    parser.add_argument("--ignore-no-grasp", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    roots = list(args.root)
    if args.run_name:
        roots.extend(discover_run_roots(args.run_name, args.search_base))
    roots = sorted(set(p.resolve() for p in roots if p.exists()))
    if not roots:
        raise FileNotFoundError("No result roots found. Use --root or --run-name.")

    trials: list[Trial] = []
    for root in roots:
        trials.extend(iter_trials(root))
    if not trials:
        raise FileNotFoundError("No trial_results.csv rows found under the selected roots.")

    print("Result roots:")
    for root in roots:
        print(f"  {root}")
    print(f"Total trials loaded: {len(trials)}")
    print(f"ignore_no_grasp: {args.ignore_no_grasp}")

    by_method = defaultdict(list)
    by_method_object_task = defaultdict(list)
    by_object_task = defaultdict(list)
    for t in trials:
        by_method[(t.method,)].append(t)
        by_method_object_task[(t.method, t.object_name, t.task)].append(t)
        by_object_task[(t.object_name, t.task)].append(t)

    print_summary("By Method", by_method, args.ignore_no_grasp)
    print_summary("By Method / Object / Task", by_method_object_task, args.ignore_no_grasp)
    print_summary("By Object / Task Across Methods", by_object_task, args.ignore_no_grasp)

    if args.output_csv:
        write_csv(args.output_csv, by_method_object_task, args.ignore_no_grasp)
        print(f"\nCSV written: {args.output_csv}")


if __name__ == "__main__":
    main()

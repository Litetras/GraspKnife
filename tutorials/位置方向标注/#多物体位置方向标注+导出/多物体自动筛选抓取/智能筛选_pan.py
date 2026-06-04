import os
import glob
import json
import yaml
import random
import re
import numpy as np
from scipy.spatial.transform import Rotation

# =====================================================================
# 🍳 Pan 专属任务导向抓取规则表
# =====================================================================
# 你的规则：
# pan / Handle / Up -> cook
# pan / Rim    / Up -> pass
#
# 注意：
# - Handle 会对应把手区域
# - Rim 会对应非 Handle 区域，也就是锅身 / 锅口区域
# =====================================================================
TASK_RULES = {
    "Pan": [
        {"task": "cook", "region": "Handle", "orientations": ["Up"]},
        {"task": "pass", "region": "Rim", "orientations": ["Up"]}
    ]
}

# =====================================================================
# 🎛️ Pan 专属角度阈值配置
# =====================================================================
CATEGORY_ANGLE_CONFIG = {
    "default": {"strict": 15.0, "relaxed": 25.0},
    "pans":    {"strict": 15.0, "relaxed": 35.0}
}


def natural_key(text):
    """
    自然排序：pan_2 排在 pan_10 前面。
    """
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def check_grasp_region(pos, info, target_region):
    """
    判断抓取点属于 Handle 还是 Rim。

    当前 Pan 的几何标注是二分：
    - Handle = 把手区域
    - Rim    = 非把手区域，也就是锅身 / 锅口区域
    """
    coord = pos[info["split_axis"]]
    is_handle_side = False

    if info["mode"] == "2_points":
        if info["target_is_positive"] and coord > info["boundary_coord"]:
            is_handle_side = True
        elif not info["target_is_positive"] and coord < info["boundary_coord"]:
            is_handle_side = True

    elif info["mode"] == "3_points":
        if info["boundary_min"] <= coord <= info["boundary_max"]:
            is_handle_side = True

    region_lower = target_region.lower()

    if region_lower == "handle":
        return is_handle_side

    elif region_lower == "rim":
        return not is_handle_side

    else:
        return not is_handle_side


def check_kinematic_constraints(rot_matrix, category_name, required_region, info):
    """
    Pan 专属运动学约束。

    对 Handle 抓取：
    - 避免夹爪闭合方向完全顺着把手方向。
    - 如果太严格，可以把 0.85 改成 0.95，或者直接 return True, "Pass"。
    """
    closing_axis = rot_matrix[:, 1]

    cat_lower = category_name.lower()
    reg_lower = required_region.lower()

    if "pan" in cat_lower and reg_lower == "handle":
        split_vec = np.zeros(3)
        split_vec[info["split_axis"]] = 1.0

        parallelism_to_handle = abs(np.dot(closing_axis, split_vec))

        if parallelism_to_handle > 0.85:
            return False, "Pan_Handle_Along_Handle_Bad_Grasp"

    return True, "Pass"


def get_pan_rules(category_name):
    """
    只匹配 Pan / pans 类别。
    """
    cat_lower = category_name.lower()

    pan_aliases = ["pan", "pans", "skillet", "frying_pan", "frying pan"]

    if any(alias in cat_lower for alias in pan_aliases):
        return TASK_RULES["Pan"]

    return []


def make_empty_stats(total_grasps):
    return {
        "total_grasps": total_grasps,
        "parse_error": 0,
        "region_match": 0,
        "orientation_pass": 0,
        "kinematic_rejected_total": 0,
        "kinematic_reasons": {},
        "accepted": 0,
        "max_dot_all": None,
        "max_dot_region": None
    }


def update_max_value(stats, key, value):
    if stats[key] is None:
        stats[key] = float(value)
    else:
        stats[key] = float(max(stats[key], value))


def collect_grasps_for_combo(
    grasps_dict,
    info,
    category_name,
    required_region,
    target_ori_name,
    orientations_dict,
    dot_threshold
):
    """
    针对一个组合，比如 Handle_Up 或 Rim_Up，完整统计筛选过程。

    返回：
    - matrix_list: 通过筛选的抓取矩阵
    - stats: 详细统计，用于诊断为什么没生成
    """
    total_grasps = len(grasps_dict)
    stats = make_empty_stats(total_grasps)
    matrix_list = []

    if target_ori_name not in orientations_dict:
        stats["missing_orientation"] = True
        return matrix_list, stats

    stats["missing_orientation"] = False

    target_vector = np.array(orientations_dict[target_ori_name]["vector"], dtype=np.float64)

    for grasp_id, grasp_data in grasps_dict.items():
        try:
            pos = np.array(grasp_data["position"], dtype=np.float64)

            w = grasp_data["orientation"]["w"]
            x, y, z = grasp_data["orientation"]["xyz"]

            rot_matrix = Rotation.from_quat([x, y, z, w]).as_matrix()
            approach_vector = rot_matrix[:, 2]

            dot_value = float(np.dot(approach_vector, target_vector))
            update_max_value(stats, "max_dot_all", dot_value)

            # 1. Region 筛选
            if not check_grasp_region(pos, info, required_region):
                continue

            stats["region_match"] += 1
            update_max_value(stats, "max_dot_region", dot_value)

            # 2. Orientation 筛选
            if dot_value < dot_threshold:
                continue

            stats["orientation_pass"] += 1

            # 3. 运动学约束筛选
            is_valid, reason = check_kinematic_constraints(
                rot_matrix,
                category_name,
                required_region,
                info
            )

            if not is_valid:
                stats["kinematic_rejected_total"] += 1
                stats["kinematic_reasons"][reason] = (
                    stats["kinematic_reasons"].get(reason, 0) + 1
                )
                continue

            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos

            matrix_list.append(T.tolist())
            stats["accepted"] += 1

        except Exception:
            stats["parse_error"] += 1
            continue

    return matrix_list, stats


def diagnose_missing_reason(strict_stats, relaxed_stats, region_name, ori_name):
    """
    根据 strict / relaxed 的统计，推断为什么没有生成 JSON。
    """
    # 优先看 relaxed，因为最终补救也是靠 relaxed
    stats = relaxed_stats

    if stats.get("missing_orientation", False):
        return (
            f"缺少方向配置: orientations 里没有 '{ori_name}'。"
            f"请检查 pan_category_grasp_directions.json 是否标了 {ori_name}。"
        )

    if stats["total_grasps"] == 0:
        return "YAML 中 grasps 为空。"

    if stats["parse_error"] >= stats["total_grasps"]:
        return "所有 grasp 数据解析失败，可能 YAML 格式或 orientation 字段异常。"

    if stats["region_match"] == 0:
        return (
            f"Region 筛选为 0。说明没有任何抓取点落在 {region_name} 区域。"
            f"大概率是 pan_dataset_boundaries_auto.json 里的 split_axis / boundary_coord / "
            f"target_is_positive 标错，或者该模型把手/锅口分割不对。"
        )

    if stats["orientation_pass"] == 0:
        max_dot_region = stats["max_dot_region"]
        max_dot_text = "None" if max_dot_region is None else f"{max_dot_region:.4f}"
        return (
            f"Orientation 筛选为 0。说明 {region_name} 区域里没有抓取的 approach_vector "
            f"接近 {ori_name}。"
            f"relaxed 下该区域最大 dot={max_dot_text}。"
            f"可能是 Up 方向标注不对，或者 relaxed 角度仍然太严格。"
        )

    if stats["kinematic_rejected_total"] > 0 and stats["accepted"] == 0:
        return (
            f"通过 Region + Orientation 的抓取全部被运动学约束拦截。"
            f"拦截原因: {stats['kinematic_reasons']}。"
            f"如果你想保留更多 Handle 抓取，可以把 "
            f"parallelism_to_handle > 0.85 改成 0.95，或者临时关闭该约束。"
        )

    if stats["accepted"] == 0:
        return (
            "最终 accepted 为 0，但未命中明确原因。"
            f"统计信息: region_match={stats['region_match']}, "
            f"orientation_pass={stats['orientation_pass']}, "
            f"kinematic_rejected={stats['kinematic_rejected_total']}, "
            f"parse_error={stats['parse_error']}。"
        )

    return "未知原因。"


def export_combo_json(
    output_dir,
    category_name,
    base_name,
    task_name,
    region_name,
    ori_name,
    matrix_list,
    target_max_grasps
):
    """
    导出单个组合的 JSON。
    """
    category_out_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_out_dir, exist_ok=True)

    if len(matrix_list) > target_max_grasps:
        matrix_list = random.sample(matrix_list, target_max_grasps)

    output_data = {
        "object": {
            "file": f"{base_name}.obj",
            "scale": 1.0
        },
        "task_semantics": {
            "task": task_name,
            "region": region_name,
            "orientation": ori_name
        },
        "grasps": {
            "transforms": matrix_list
        }
    }

    output_filename = f"{base_name}_{region_name}_{ori_name}.json"
    output_path = os.path.join(category_out_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    return output_path, len(matrix_list)


def filter_and_convert_pan_grasps(dataset_json_path, yaml_dir, output_dir):
    print("=" * 70)
    print("🍳 启动 Pan 专属任务导向抓取清洗器 + 缺失原因诊断版")
    print("🛡️ 规则: 仅导出符合 [Region + Orientation] 约束的 Pan 抓取")
    print("🎯 当前任务规则:")
    print("   1. cook -> Handle / Up")
    print("   2. pass -> Rim    / Up")
    print("=" * 70)

    TARGET_MAX_GRASPS = 400
    MIN_GRASPS_THRESHOLD = 50

    if not os.path.exists(dataset_json_path):
        print(f"❌ 找不到 Pan 数据集总表: {dataset_json_path}")
        return

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    yaml_files = glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)

    if not yaml_files:
        print(f"❌ 在 YAML 输入目录中没有找到任何 yaml 文件: {yaml_dir}")
        return

    yaml_map = {}
    for yaml_path in yaml_files:
        base_name = os.path.basename(yaml_path).replace(".yaml", "")
        yaml_map[base_name] = yaml_path

    processed_count = 0
    total_grasps_saved = 0
    exported_json_count = 0
    expected_combo_count = 0
    missing_combo_count = 0

    skipped_no_yaml = 0
    skipped_no_rule = 0
    skipped_no_orientation = 0
    skipped_no_grasps = 0

    extra_yaml_not_in_dataset = len([name for name in yaml_map if name not in dataset_info])

    debug_report = {
        "config": {
            "dataset_json": dataset_json_path,
            "yaml_dir": yaml_dir,
            "output_dir": output_dir,
            "target_max_grasps": TARGET_MAX_GRASPS,
            "min_grasps_threshold": MIN_GRASPS_THRESHOLD,
            "category_angle_config": CATEGORY_ANGLE_CONFIG,
            "task_rules": TASK_RULES
        },
        "summary": {},
        "missing_combos": [],
        "skipped_models": [],
        "per_model": {}
    }

    dataset_base_names = sorted(dataset_info.keys(), key=natural_key)

    for base_name in dataset_base_names:
        info = dataset_info[base_name]
        category_name = info.get("category", "unknown")

        category_rules = get_pan_rules(category_name)

        if not category_rules:
            skipped_no_rule += 1
            debug_report["skipped_models"].append({
                "base_name": base_name,
                "reason": f"无 Pan 规则，category={category_name}"
            })
            continue

        if base_name not in yaml_map:
            skipped_no_yaml += 1
            debug_report["skipped_models"].append({
                "base_name": base_name,
                "reason": "dataset 中有该实例，但 grasps 目录里找不到对应 YAML"
            })
            print(f"❌ 跳过 {base_name}: 找不到对应 YAML")
            continue

        orientations_dict = info.get("orientations", info.get("intents", {}))

        if not orientations_dict:
            skipped_no_orientation += 1
            debug_report["skipped_models"].append({
                "base_name": base_name,
                "reason": "缺少 orientations / intents"
            })
            print(f"⚠️ 跳过 {base_name}: 缺少 orientations / intents")
            continue

        angle_cfg = CATEGORY_ANGLE_CONFIG.get(
            category_name,
            CATEGORY_ANGLE_CONFIG["default"]
        )

        dot_threshold = np.cos(np.radians(angle_cfg["strict"]))
        relaxed_dot_threshold = np.cos(np.radians(angle_cfg["relaxed"]))

        yaml_path = yaml_map[base_name]

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            yaml_data = {}

        grasps_dict = yaml_data.get("grasps", {})

        if not grasps_dict:
            skipped_no_grasps += 1
            debug_report["skipped_models"].append({
                "base_name": base_name,
                "reason": "YAML 中没有 grasps 数据"
            })
            print(f"⚠️ 跳过 {base_name}: YAML 中没有 grasps 数据")
            continue

        processed_count += 1

        debug_report["per_model"][base_name] = {
            "category": category_name,
            "yaml_path": yaml_path,
            "total_yaml_grasps": len(grasps_dict),
            "combos": {}
        }

        print("\n" + "-" * 70)
        print(f"🔍 正在检查 [{processed_count}] {base_name} | category={category_name}")
        print(f"   YAML grasps 数量: {len(grasps_dict)}")
        print(f"   strict={angle_cfg['strict']}°, relaxed={angle_cfg['relaxed']}°")

        # =====================================================================
        # 对每个预期组合都强制检查：
        # Handle_Up 和 Rim_Up 都会被检查，哪怕 strict 是 0。
        # =====================================================================
        for rule in category_rules:
            task_name = rule["task"]
            region_name = rule["region"]

            for ori_name in rule["orientations"]:
                expected_combo_count += 1
                combo_label = f"{region_name}_{ori_name}"

                strict_list, strict_stats = collect_grasps_for_combo(
                    grasps_dict=grasps_dict,
                    info=info,
                    category_name=category_name,
                    required_region=region_name,
                    target_ori_name=ori_name,
                    orientations_dict=orientations_dict,
                    dot_threshold=dot_threshold
                )

                # strict 数量不足时，无论 strict 是不是 0，都强制跑 relaxed
                used_stage = "strict"
                final_list = strict_list
                relaxed_list = []
                relaxed_stats = None

                if len(strict_list) < MIN_GRASPS_THRESHOLD:
                    relaxed_list, relaxed_stats = collect_grasps_for_combo(
                        grasps_dict=grasps_dict,
                        info=info,
                        category_name=category_name,
                        required_region=region_name,
                        target_ori_name=ori_name,
                        orientations_dict=orientations_dict,
                        dot_threshold=relaxed_dot_threshold
                    )

                    used_stage = "relaxed"
                    final_list = relaxed_list

                    print(
                        f"   🔄 [动态放宽] {base_name} ({combo_label}) "
                        f"strict={len(strict_list)} < {MIN_GRASPS_THRESHOLD}，"
                        f"按 {angle_cfg['relaxed']}° 放宽后: {len(relaxed_list)}"
                    )

                if relaxed_stats is None:
                    # 如果没有跑 relaxed，也保存一个空位，方便报告结构统一
                    relaxed_stats = {
                        "not_run": True
                    }

                combo_report = {
                    "task": task_name,
                    "region": region_name,
                    "orientation": ori_name,
                    "used_stage": used_stage,
                    "strict": strict_stats,
                    "relaxed": relaxed_stats,
                    "final_count_before_sampling": len(final_list),
                    "exported": False,
                    "output_path": None,
                    "missing_reason": None
                }

                # 如果 final 为 0，不导出，但打印原因并写入报告
                if len(final_list) == 0:
                    missing_combo_count += 1

                    reason = diagnose_missing_reason(
                        strict_stats=strict_stats,
                        relaxed_stats=relaxed_stats if "not_run" not in relaxed_stats else strict_stats,
                        region_name=region_name,
                        ori_name=ori_name
                    )

                    combo_report["missing_reason"] = reason

                    missing_item = {
                        "base_name": base_name,
                        "category": category_name,
                        "combo": combo_label,
                        "task": task_name,
                        "region": region_name,
                        "orientation": ori_name,
                        "reason": reason,
                        "strict_stats": strict_stats,
                        "relaxed_stats": relaxed_stats
                    }

                    debug_report["missing_combos"].append(missing_item)

                    print(f"   ❌ 未生成: {base_name}_{combo_label}.json")
                    print(f"      原因: {reason}")
                    print(
                        f"      strict统计: region={strict_stats['region_match']}, "
                        f"ori_pass={strict_stats['orientation_pass']}, "
                        f"killed={strict_stats['kinematic_rejected_total']}, "
                        f"accepted={strict_stats['accepted']}, "
                        f"max_dot_region={strict_stats['max_dot_region']}"
                    )

                    if "not_run" not in relaxed_stats:
                        print(
                            f"      relaxed统计: region={relaxed_stats['region_match']}, "
                            f"ori_pass={relaxed_stats['orientation_pass']}, "
                            f"killed={relaxed_stats['kinematic_rejected_total']}, "
                            f"accepted={relaxed_stats['accepted']}, "
                            f"max_dot_region={relaxed_stats['max_dot_region']}"
                        )

                    debug_report["per_model"][base_name]["combos"][combo_label] = combo_report
                    continue

                # 有结果则导出
                output_path, saved_count = export_combo_json(
                    output_dir=output_dir,
                    category_name=category_name,
                    base_name=base_name,
                    task_name=task_name,
                    region_name=region_name,
                    ori_name=ori_name,
                    matrix_list=final_list,
                    target_max_grasps=TARGET_MAX_GRASPS
                )

                total_grasps_saved += saved_count
                exported_json_count += 1

                combo_report["exported"] = True
                combo_report["output_path"] = output_path
                combo_report["saved_count"] = saved_count

                debug_report["per_model"][base_name]["combos"][combo_label] = combo_report

                print(
                    f"   ✅ 导出: {os.path.basename(output_path)} "
                    f"(保存 {saved_count} 个 Pan 抓取, 使用 {used_stage})"
                )

    # =====================================================================
    # 保存 debug 报告
    # =====================================================================
    debug_report["summary"] = {
        "processed_models": processed_count,
        "expected_combo_count": expected_combo_count,
        "exported_json_count": exported_json_count,
        "missing_combo_count": missing_combo_count,
        "total_grasps_saved": total_grasps_saved,
        "skipped_no_yaml": skipped_no_yaml,
        "skipped_no_rule": skipped_no_rule,
        "skipped_no_orientation": skipped_no_orientation,
        "skipped_no_grasps": skipped_no_grasps,
        "extra_yaml_not_in_dataset": extra_yaml_not_in_dataset
    }

    report_path = os.path.join(output_dir, "pan_grasp_filter_debug_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(debug_report, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 Pan 抓取清洗完毕！")
    print(f"✅ 共处理 Pan 模型: {processed_count} 个")
    print(f"🎯 理论应检查组合数: {expected_combo_count} 个")
    print(f"✅ 实际导出 JSON 数: {exported_json_count} 个")
    print(f"❌ 未生成组合数: {missing_combo_count} 个")
    print(f"✅ 总共导出高质量抓取: {total_grasps_saved} 个")
    print(f"⏭️ dataset 中有但找不到 YAML: {skipped_no_yaml} 个")
    print(f"⏭️ YAML 不在 Pan dataset 中: {extra_yaml_not_in_dataset} 个")
    print(f"⏭️ 无 Pan 规则: {skipped_no_rule} 个")
    print(f"⏭️ 缺少 orientations: {skipped_no_orientation} 个")
    print(f"⏭️ YAML 无 grasps: {skipped_no_grasps} 个")
    print(f"📁 Pan 抓取 JSON 已保存在: {output_dir}")
    print(f"🧾 Debug 报告已保存至: {report_path}")

    if missing_combo_count > 0:
        print("\n❌ 未生成抓取的组合如下：")
        for item in debug_report["missing_combos"]:
            print(
                f"   - {item['base_name']}_{item['combo']}.json "
                f"| task={item['task']} | 原因: {item['reason']}"
            )

    print("=" * 70)


if __name__ == "__main__":
    DATASET_ROOT = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集"

    DATASET_JSON = "/home/zyp/GraspGen/final_pan_task_oriented_dataset.json"

    YAML_INPUT_DIR = os.path.join(DATASET_ROOT, "grasps")

    JSON_OUTPUT_DIR = os.path.join(DATASET_ROOT, "task_oriented_grasps_json_pan")

    filter_and_convert_pan_grasps(
        DATASET_JSON,
        YAML_INPUT_DIR,
        JSON_OUTPUT_DIR
    )
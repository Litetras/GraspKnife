import os
import glob
import json
import yaml
import random
import numpy as np
from scipy.spatial.transform import Rotation

# =====================================================================
# ☕ Mug 专属任务导向抓取规则表
# =====================================================================
TASK_RULES = {
    "Mug": [
        {"task": "Mug pouring", "region": "Handle", "orientations": ["Up"]},
        {"task": "Mug passing", "region": "Rim", "orientations": ["Front"]}
    ]
}

# =====================================================================
# 🎛️ Mug 专属角度阈值配置
# =====================================================================
CATEGORY_ANGLE_CONFIG = {
    "default": {"strict": 15.0, "relaxed": 25.0},
    "mugs": {"strict": 15.0, "relaxed": 25.0},
    "7_mugs": {"strict": 15.0, "relaxed": 25.0}
}


def check_grasp_region(pos, info, target_region):
    """
    判断抓取点属于 Handle 还是 Rim。

    当前 Mug 的几何标注是二分：
    - Handle = 把手区域
    - Rim    = 非把手区域，也就是杯口 / 杯身区域
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
    else:
        # Rim / Body / Head 等都视为非 Handle 区域
        return not is_handle_side


def check_kinematic_constraints(rot_matrix, category_name, required_region, info):
    """
    Mug 专属运动学约束。

    对 Mug Handle：
    - 强制夹爪闭合轴接近世界 Y 轴
    - 主要用于减少穿模、斜夹、从奇怪方向夹把手的问题
    """
    closing_axis = rot_matrix[:, 1]  # 夹爪局部 Y 轴，通常是手指开合方向

    cat_lower = category_name.lower()
    reg_lower = required_region.lower()

    if "mug" in cat_lower and reg_lower == "handle":
        world_y = np.array([0.0, 1.0, 0.0])
        parallelism_to_y = abs(np.dot(closing_axis, world_y))

        if parallelism_to_y < 0.60:
            return False, "Mug_Z_Axis_Collision"

    return True, "Pass"


def get_mug_rules(category_name):
    """
    只匹配 Mug / mugs / cup 类别。
    """
    cat_lower = category_name.lower()

    mug_aliases = ["mug", "mugs", "cup", "7_mugs"]

    if any(alias in cat_lower for alias in mug_aliases):
        return TASK_RULES["Mug"]

    return []


def filter_and_convert_mug_grasps(dataset_json_path, yaml_dir, output_dir):
    print("=" * 70)
    print("☕ 启动 Mug 专属任务导向抓取清洗器")
    print("🛡️ 规则: 仅导出符合 [Region + Orientation] 约束的 Mug 抓取")
    print("🎯 当前任务规则:")
    print("   1. Mug pouring -> Handle / Up")
    print("   2. Mug passing -> Rim    / Front")
    print("🎯 Mug Handle 约束: 夹爪闭合轴接近世界 Y 轴，减少穿模坏抓取")
    print("=" * 70)

    TARGET_MAX_GRASPS = 400
    MIN_GRASPS_THRESHOLD = 50

    if not os.path.exists(dataset_json_path):
        print(f"❌ 找不到数据集总表: {dataset_json_path}")
        return

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    yaml_files = glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)

    if not yaml_files:
        print(f"❌ 在 YAML 输入目录中没有找到任何 yaml 文件: {yaml_dir}")
        return

    processed_count = 0
    total_grasps_saved = 0

    skipped_not_in_dataset = 0
    skipped_no_rule = 0
    skipped_no_orientation = 0
    skipped_no_grasps = 0

    for yaml_path in yaml_files:
        base_name = os.path.basename(yaml_path).replace(".yaml", "")

        if base_name not in dataset_info:
            skipped_not_in_dataset += 1
            continue

        info = dataset_info[base_name]
        category_name = info.get("category", "unknown")

        category_rules = get_mug_rules(category_name)

        if not category_rules:
            skipped_no_rule += 1
            continue

        orientations_dict = info.get("orientations", info.get("intents", {}))

        if not orientations_dict:
            print(f"⚠️ 跳过 {base_name}: 缺少 orientations / intents")
            skipped_no_orientation += 1
            continue

        angle_cfg = CATEGORY_ANGLE_CONFIG.get(
            category_name,
            CATEGORY_ANGLE_CONFIG["default"]
        )

        dot_threshold = np.cos(np.radians(angle_cfg["strict"]))
        relaxed_dot_threshold = np.cos(np.radians(angle_cfg["relaxed"]))

        category_out_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_out_dir, exist_ok=True)

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            yaml_data = {}

        grasps_dict = yaml_data.get("grasps", {})

        if not grasps_dict:
            skipped_no_grasps += 1
            continue

        valid_grasps_by_combo = {}

        debug_kills = {
            "Mug_Z_Axis_Collision": 0
        }

        # =====================================================================
        # 第一轮：strict 阈值筛选
        # =====================================================================
        for grasp_id, grasp_data in grasps_dict.items():
            pos = np.array(grasp_data["position"])

            w = grasp_data["orientation"]["w"]
            x, y, z = grasp_data["orientation"]["xyz"]

            rot_matrix = Rotation.from_quat([x, y, z, w]).as_matrix()
            approach_vector = rot_matrix[:, 2]

            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos
            T_list = T.tolist()

            for rule in category_rules:
                task_name = rule["task"]
                required_region = rule["region"]
                allowed_orientations = rule["orientations"]

                if not check_grasp_region(pos, info, required_region):
                    continue

                for target_ori_name in allowed_orientations:
                    if target_ori_name not in orientations_dict:
                        continue

                    target_vector = np.array(
                        orientations_dict[target_ori_name]["vector"]
                    )

                    if np.dot(approach_vector, target_vector) >= dot_threshold:
                        is_valid, reason = check_kinematic_constraints(
                            rot_matrix,
                            category_name,
                            required_region,
                            info
                        )

                        if not is_valid:
                            debug_kills[reason] += 1
                            continue

                        combo_key = (
                            task_name,
                            required_region,
                            target_ori_name
                        )

                        if combo_key not in valid_grasps_by_combo:
                            valid_grasps_by_combo[combo_key] = []

                        valid_grasps_by_combo[combo_key].append(T_list)

        # =====================================================================
        # 第二轮：数量不足时 relaxed 阈值补救
        # =====================================================================
        for combo_key, matrix_list in valid_grasps_by_combo.items():
            task_name, region_name, ori_name = combo_key

            if len(matrix_list) < MIN_GRASPS_THRESHOLD:
                relaxed_list = []

                if ori_name not in orientations_dict:
                    continue

                target_vector = np.array(
                    orientations_dict[ori_name]["vector"]
                )

                for grasp_id, grasp_data in grasps_dict.items():
                    pos = np.array(grasp_data["position"])

                    if not check_grasp_region(pos, info, region_name):
                        continue

                    w = grasp_data["orientation"]["w"]
                    x, y, z = grasp_data["orientation"]["xyz"]

                    rot_matrix = Rotation.from_quat([x, y, z, w]).as_matrix()
                    approach_vector = rot_matrix[:, 2]

                    if np.dot(approach_vector, target_vector) >= relaxed_dot_threshold:
                        is_valid, reason = check_kinematic_constraints(
                            rot_matrix,
                            category_name,
                            region_name,
                            info
                        )

                        if not is_valid:
                            debug_kills[reason] += 1
                            continue

                        T = np.eye(4)
                        T[:3, :3] = rot_matrix
                        T[:3, 3] = pos

                        relaxed_list.append(T.tolist())

                matrix_list = relaxed_list

                print(
                    f"   🔄 [动态放宽] {base_name} "
                    f"({region_name}_{ori_name}) 数量不足，"
                    f"按 {angle_cfg['relaxed']}° 放宽后提升至: {len(matrix_list)}"
                )

            if len(matrix_list) > TARGET_MAX_GRASPS:
                matrix_list = random.sample(matrix_list, TARGET_MAX_GRASPS)

            if len(matrix_list) == 0:
                continue

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

            total_grasps_saved += len(matrix_list)

            print(
                f"✅ 导出: [{category_name}] {output_filename} "
                f"(保存 {len(matrix_list)} 个 Mug 抓取)"
            )

        if debug_kills["Mug_Z_Axis_Collision"] > 0:
            print(
                f"   🚨 [Debug] 在 {base_name} 中拦截了 "
                f"{debug_kills['Mug_Z_Axis_Collision']} "
                f"个 Mug Handle 干涉 / 斜夹坏抓取"
            )

        processed_count += 1

    print("\n" + "=" * 70)
    print("🎉 Mug 抓取清洗完毕！")
    print(f"✅ 共处理 Mug 模型: {processed_count} 个")
    print(f"✅ 总共导出高质量抓取: {total_grasps_saved} 个")
    print(f"⏭️ YAML 不在 dataset 中，跳过: {skipped_not_in_dataset} 个")
    print(f"⏭️ 无 Mug 规则，跳过: {skipped_no_rule} 个")
    print(f"⏭️ 缺少 orientations，跳过: {skipped_no_orientation} 个")
    print(f"⏭️ YAML 无 grasps，跳过: {skipped_no_grasps} 个")
    print(f"📁 Mug 抓取 JSON 已保存在: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    DATASET_ROOT = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集"

    DATASET_JSON = "/home/zyp/GraspGen/final_task_oriented_dataset.json"

    YAML_INPUT_DIR = os.path.join(DATASET_ROOT, "grasps")

    JSON_OUTPUT_DIR = os.path.join(DATASET_ROOT, "task_oriented_grasps_json_mug")

    filter_and_convert_mug_grasps(
        DATASET_JSON,
        YAML_INPUT_DIR,
        JSON_OUTPUT_DIR
    )
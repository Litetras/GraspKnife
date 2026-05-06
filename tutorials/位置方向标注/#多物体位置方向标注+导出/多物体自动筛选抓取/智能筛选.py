import os
import glob
import json
import yaml
import random
import numpy as np
from scipy.spatial.transform import Rotation

# =====================================================================
# 🌟 核心规则注入：顶会级任务导向抓取规则表
# =====================================================================
TASK_RULES = {
    "Knife": [
        {"task": "Knife cutting", "region": "Handle", "orientations": ["Up"]},
        {"task": "Knife passing", "region": "Blade", "orientations": ["Down"]}
    ],
    "Hammer": [
        {"task": "Hammer striking", "region": "Handle", "orientations": ["Up"]},
        {"task": "Hammer pulling", "region": "Handle", "orientations": ["Down"]},
        {"task": "Hammer passing", "region": "Head", "orientations": ["Up", "Down"]}
    ],
    "Brush": [
        {"task": "Brush cleaning", "region": "Handle", "orientations": ["Up"]},
        {"task": "Brush passing", "region": "Head", "orientations": ["Up", "Down"]}
    ],
    "Mug": [
        {"task": "Mug pouring", "region": "Handle", "orientations": ["Up"]},
        {"task": "Mug passing", "region": "Rim", "orientations": ["Front"]}
    ],
    "Drill": [
        {"task": "Drill operation", "region": "Handle", "orientations": ["Up"]},
        {"task": "Drill passing", "region": "Head", "orientations": ["Left", "Right", "Front"]}
    ],
    "Screwdriver": [
        {"task": "Screwdriver driving", "region": "Handle", "orientations": ["Up"]},
        {"task": "Screwdriver passing", "region": "Shaft", "orientations": ["Left", "Right", "Front", "Back"]}
    ],
    "Spoon": [
        {"task": "Spoon scooping", "region": "Handle", "orientations": ["Up"]},
        {"task": "Spoon passing", "region": "Head", "orientations": ["Left", "Right", "Front"]}
    ]
}

# =====================================================================
# 🎛️ 动态类别角度阈值配置 (核心升级！)
# =====================================================================
CATEGORY_ANGLE_CONFIG = {
    "default":        {"strict": 15.0, "relaxed": 25.0},
    "6_screwdrivers": {"strict": 35.0, "relaxed": 45.0}, # 🪛 螺丝刀专属放宽：允许从斜后方抓侧面，不要全挤在屁股上
    "5_spoons":       {"strict": 25.0, "relaxed": 35.0}  # 🥄 勺子专属放宽：因为下面加了严苛的Roll限制，需放宽Z轴以保证数量
}

def check_grasp_region(pos, info, target_region):
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
            
    if target_region.lower() == "handle":
        return is_handle_side
    else:
        return not is_handle_side

def check_kinematic_constraints(rot_matrix, category_name, required_region, info):
    """
    🎯 运动学硬约束收容所：处理各种奇怪的乱抓现象
    """
    closing_axis = rot_matrix[:, 1] # 夹爪局部 Y 轴 (手指开合方向)
    
    cat_lower = category_name.lower()
    reg_lower = required_region.lower()

    # 1. ☕ Mug Handle 专杀：强制要求夹爪闭合轴高度平行于绿色 Y 轴
    if "mug" in cat_lower and reg_lower == "handle":
        world_y = np.array([0.0, 1.0, 0.0])
        parallelism_to_y = abs(np.dot(closing_axis, world_y))
        if parallelism_to_y < 0.60:
            return False, "Mug_Z_Axis_Collision"

    # 2. 🥄 Spoon Handle 专杀：强制要求横向抓取，解决“杂乱斜捏”问题！
    if "spoon" in cat_lower and reg_lower == "handle":
        split_vec = np.zeros(3)
        split_vec[info["split_axis"]] = 1.0 # 也就是勺子的长轴 (X轴)
        
        # 必须横向跨过勺柄！即闭合轴必须垂直于勺子的长轴。
        # parallelism 越接近 0 越垂直，如果 > 0.40 说明夹爪斜着捏或者顺着捏了，必须杀掉！
        parallelism_to_x = abs(np.dot(closing_axis, split_vec))
        if parallelism_to_x > 0.40:
            return False, "Spoon_Diagonal_Messy_Grasp"
            
    return True, "Pass"

def filter_and_convert_grasps(dataset_json_path, yaml_dir, output_dir):
    print("="*70)
    print("🧹 启动【顶会级任务导向】终极抓取清洗器 (动态阈值+姿态防呆版)")
    print("🛡️ 规则: 仅导出符合 [Region + Orientation] 约束的数据")
    print("🎯 特殊1: Mug Handle 锁定绝对Y轴，防穿模！")
    print("🎯 特殊2: Spoon Handle 锁定正交X轴，防杂乱斜捏！")
    print("🎯 特殊3: 螺丝刀与勺子应用独立超宽容差，拯救稀缺分布！")
    print("="*70)

    TARGET_MAX_GRASPS = 400
    MIN_GRASPS_THRESHOLD = 50

    if not os.path.exists(dataset_json_path):
        print(f"❌ 找不到数据集总表: {dataset_json_path}")
        return
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    yaml_files = glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)
    if not yaml_files:
        return

    processed_count = 0
    total_grasps_saved = 0

    for yaml_path in yaml_files:
        base_name = os.path.basename(yaml_path).replace(".yaml", "")
        
        if base_name not in dataset_info:
            continue
            
        info = dataset_info[base_name]
        category_name = info.get("category", "unknown")
        
        # 动态获取当前类别的角度阈值
        angle_cfg = CATEGORY_ANGLE_CONFIG.get(category_name, CATEGORY_ANGLE_CONFIG["default"])
        dot_threshold = np.cos(np.radians(angle_cfg["strict"]))
        relaxed_dot_threshold = np.cos(np.radians(angle_cfg["relaxed"]))
        
        category_rules = []
        cat_lower = category_name.lower()
        
        ALIASES = {
            "Knife": ["knife", "knives"],
            "Hammer": ["hammer", "mallet"],
            "Brush": ["brush"],
            "Mug": ["mug", "cup"],
            "Drill": ["drill"],
            "Screwdriver": ["screwdriver"],
            "Spoon": ["spoon", "ladle", "scoop"]
        }
        
        for rule_key, rules in TASK_RULES.items():
            keywords = ALIASES.get(rule_key, [rule_key.lower()])
            if any(kw in cat_lower for kw in keywords):
                category_rules = rules
                break
        
        if not category_rules:
            continue

        orientations_dict = info.get("orientations", info.get("intents", {}))
        if not orientations_dict:
            continue

        category_out_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_out_dir, exist_ok=True)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        grasps_dict = yaml_data.get("grasps", {})
        if not grasps_dict:
            continue

        valid_grasps_by_combo = {}
        
        # 🌟 专属 Debug 统计器
        debug_kills = {"Mug_Z_Axis_Collision": 0, "Spoon_Diagonal_Messy_Grasp": 0}

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
                        
                    target_vector = np.array(orientations_dict[target_ori_name]["vector"])
                    
                    if np.dot(approach_vector, target_vector) >= dot_threshold:
                        
                        # ================= 🚀 多模态硬约束拦截 =================
                        is_valid, reason = check_kinematic_constraints(rot_matrix, category_name, required_region, info)
                        if not is_valid:
                            debug_kills[reason] += 1
                            continue
                        # ======================================================
                        
                        combo_key = (task_name, required_region, target_ori_name)
                        if combo_key not in valid_grasps_by_combo:
                            valid_grasps_by_combo[combo_key] = []
                        valid_grasps_by_combo[combo_key].append(T_list)

        for combo_key, matrix_list in valid_grasps_by_combo.items():
            task_name, region_name, ori_name = combo_key
            
            if len(matrix_list) < MIN_GRASPS_THRESHOLD:
                relaxed_list = []
                target_vector = np.array(orientations_dict[ori_name]["vector"])
                
                for grasp_id, grasp_data in grasps_dict.items():
                    pos = np.array(grasp_data["position"])
                    if not check_grasp_region(pos, info, region_name):
                        continue
                    
                    w, x, y, z = grasp_data["orientation"]["w"], grasp_data["orientation"]["xyz"][0], grasp_data["orientation"]["xyz"][1], grasp_data["orientation"]["xyz"][2]
                    rot_matrix = Rotation.from_quat([x, y, z, w]).as_matrix()
                    approach_vector = rot_matrix[:, 2]
                    
                    if np.dot(approach_vector, target_vector) >= relaxed_dot_threshold:
                        is_valid, reason = check_kinematic_constraints(rot_matrix, category_name, region_name, info)
                        if not is_valid:
                            debug_kills[reason] += 1
                            continue
                            
                        T = np.eye(4); T[:3, :3] = rot_matrix; T[:3, 3] = pos
                        relaxed_list.append(T.tolist())
                
                matrix_list = relaxed_list
                print(f"   🔄 [动态放宽] {base_name} ({region_name}_{ori_name}) 数量不足，按 {angle_cfg['relaxed']}° 放宽后提升至: {len(matrix_list)}")

            if len(matrix_list) > TARGET_MAX_GRASPS:
                matrix_list = random.sample(matrix_list, TARGET_MAX_GRASPS)
                
            if len(matrix_list) == 0:
                continue
            
            output_data = {
                "object": {"file": f"{base_name}.obj", "scale": 1.0},
                "task_semantics": {"task": task_name, "region": region_name, "orientation": ori_name},
                "grasps": {"transforms": matrix_list}
            }
            
            output_filename = f"{base_name}_{region_name}_{ori_name}.json"
            output_path = os.path.join(category_out_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            total_grasps_saved += len(matrix_list)
            print(f"✅  导出: [{category_name}] {output_filename} (保存 {len(matrix_list)} 个最优抓取)")
            
        # 打印 Debug 报告
        if debug_kills["Mug_Z_Axis_Collision"] > 0:
            print(f"   🚨 [Debug] 在 {base_name} 中拦截了 {debug_kills['Mug_Z_Axis_Collision']} 个Mug干涉坏抓取！")
        if debug_kills["Spoon_Diagonal_Messy_Grasp"] > 0:
            print(f"   🚨 [Debug] 在 {base_name} 中拦截了 {debug_kills['Spoon_Diagonal_Messy_Grasp']} 个Spoon杂乱斜向抓取！")
            
        processed_count += 1

    print("\n" + "="*70)
    print(f"🎉 数据集清洗完毕！共处理 {processed_count} 个模型，导出 {total_grasps_saved} 个高质量抓取。")
    print(f"📁 数据集已保存在: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    DATASET_JSON = "final_task_oriented_dataset.json"  
    YAML_INPUT_DIR = "/home/zyp/Desktop/grasps"  
    JSON_OUTPUT_DIR = "/home/zyp/Desktop/task_oriented_grasps_json"  
    filter_and_convert_grasps(DATASET_JSON, YAML_INPUT_DIR, JSON_OUTPUT_DIR)
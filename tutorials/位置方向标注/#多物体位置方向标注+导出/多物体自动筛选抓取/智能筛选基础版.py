import os
import glob
import json
import yaml
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

def check_grasp_region(pos, info, target_region):
    """
    智能区域判断器：根据 JSON 里的边界判断抓取点是否在目标区域。
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
            
    if target_region.lower() == "handle":
        return is_handle_side
    else:
        return not is_handle_side

def filter_and_convert_grasps(dataset_json_path, yaml_dir, output_dir, angle_threshold_deg=30.0):
    print("="*70)
    print("🧹 启动【顶会级任务导向】终极抓取清洗器 (已修复模糊匹配版)")
    print("🛡️ 规则: 仅导出符合 [Region + Orientation] 约束的数据")
    print(f"📐 方向容差角度: ±{angle_threshold_deg}°")
    print("="*70)

    if not os.path.exists(dataset_json_path):
        print(f"❌ 找不到数据集总表: {dataset_json_path}")
        return
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    dot_threshold = np.cos(np.radians(angle_threshold_deg))

    yaml_files = glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)
    if not yaml_files:
        print(f"⚠️ 在 {yaml_dir} 找不到任何 YAML 文件！")
        return

    processed_count = 0
    total_grasps_saved = 0

    for yaml_path in yaml_files:
        base_name = os.path.basename(yaml_path).replace(".yaml", "")
        
        if base_name not in dataset_info:
            print(f"⏭️  [跳过] '{base_name}' 不在 JSON 总表中。")
            continue
            
        info = dataset_info[base_name]
        category_name = info.get("category", "unknown")
        
# ================= 核心修复点：智能模糊匹配类别名 =================
        category_rules = []
        cat_lower = category_name.lower()
        
        # 建立一个别名映射表，解决英语单复数不规则变化和同义词问题
        ALIASES = {
            "Knife": ["knife", "knives"],
            "Hammer": ["hammer", "mallet"],
            "Brush": ["brush"],
            "Mug": ["mug", "cup"],       # 万一你的数据集里有 cup，也能算作 mug
            "Drill": ["drill"],
            "Screwdriver": ["screwdriver"],
            "Spoon": ["spoon", "ladle", "scoop"] # 各种勺子都能兼容
        }
        
        for rule_key, rules in TASK_RULES.items():
            # 取出该规则的所有有效关键词
            keywords = ALIASES.get(rule_key, [rule_key.lower()])
            # 只要类别名中包含任何一个关键词，就算匹配成功
            if any(kw in cat_lower for kw in keywords):
                category_rules = rules
                break
        # ==============================================================
        if not category_rules:
            print(f"⏭️  [跳过] '{base_name}' (类别: {category_name}) 不在顶会规则表 TASK_RULES 中。")
            continue

        # 注意：你在生成 JSON 的时候把 `orientations` 写成了 `orientations` 还是 `intents`？
        # 兼容你的两种可能写法：
        orientations_dict = info.get("orientations", info.get("intents", {}))
        if not orientations_dict:
            print(f"⚠️  [跳过] '{base_name}' 没有操作者语义方向配置 (缺失 orientations 字段)。")
            continue

        category_out_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_out_dir, exist_ok=True)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        grasps_dict = yaml_data.get("grasps", {})
        if not grasps_dict:
            print(f"⏭️  [跳过] '{base_name}' 的 YAML 文件中没有 grasps 数据。")
            continue

        valid_grasps_by_combo = {}

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
                    # 获取我们在前面标注的该方向的物理向量
                    if target_ori_name not in orientations_dict:
                        continue 
                        
                    target_vector = np.array(orientations_dict[target_ori_name]["vector"])
                    
                    similarity = np.dot(approach_vector, target_vector)
                    if similarity >= dot_threshold:
                        combo_key = (task_name, required_region, target_ori_name)
                        if combo_key not in valid_grasps_by_combo:
                            valid_grasps_by_combo[combo_key] = []
                        valid_grasps_by_combo[combo_key].append(T_list)

        if not valid_grasps_by_combo:
            print(f"➖  [无结果] '{base_name}' 的所有抓取都不满足 Region & Orientation 约束。")

        for combo_key, matrix_list in valid_grasps_by_combo.items():
            task_name, region_name, ori_name = combo_key
            
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
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            total_grasps_saved += len(matrix_list)
            print(f"✅  导出: [{category_name}] {output_filename} (包含 {len(matrix_list)} 个最优抓取)")
            
        processed_count += 1

    print("\n" + "="*70)
    print(f"🎉 严格清洗完毕！共处理 {processed_count} 个模型，导出 {total_grasps_saved} 个高质量抓取。")
    print(f"📁 数据集已保存在: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    DATASET_JSON = "final_task_oriented_dataset.json"  
    YAML_INPUT_DIR = "/home/zyp/Desktop/grasps"  
    JSON_OUTPUT_DIR = "/home/zyp/Desktop/task_oriented_grasps_json"  
    
    ANGLE_TOLERANCE = 15.0 
    
    filter_and_convert_grasps(DATASET_JSON, YAML_INPUT_DIR, JSON_OUTPUT_DIR, ANGLE_TOLERANCE)
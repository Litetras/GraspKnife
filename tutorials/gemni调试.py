import json
import os

json_path = '/home/zyp/Desktop/zyp_dataset/tutorial/tutorial_grasp_dataset/map_uuid_to_path.json'
base_dir = '/home/zyp/Desktop/zyp_dataset/tutorial/tutorial_grasp_dataset'

with open(json_path, 'r') as f:
    mapping = json.load(f)

for obj_name in ['knife_geo_up.obj', 'knife_geo_down.obj']:
    print(f"\n--- 检查对象: {obj_name} ---")
    relative_path = mapping.get(obj_name)
    
    if not relative_path:
        print(f"❌ JSON 映射中找不到该对象")
        continue
    
    # 拼接出完整路径 (注意这里要模拟你运行环境的真实路径)
    full_path = os.path.join(base_dir, relative_path)
    print(f"映射指向的文件路径: {full_path}")
    
    if os.path.exists(full_path):
        print(f"✅ 文件物理存在")
        try:
            with open(full_path, 'r') as gf:
                grasp_data = json.load(gf)
                # 检查抓取数据是否为空，或者是否包含预期的键（如 'grasps'）
                if isinstance(grasp_data, dict):
                    print(f"✅ JSON 加载成功，包含键: {list(grasp_data.keys())}")
                elif isinstance(grasp_data, list):
                    print(f"✅ JSON 加载成功，是一个列表，长度为: {len(grasp_data)}")
        except Exception as e:
            print(f"❌ 文件存在但读取失败: {e}")
    else:
        print(f"❌ 错误: 文件在路径下不存在！请检查 JSON 里的路径是否写死成了宿主机路径。")
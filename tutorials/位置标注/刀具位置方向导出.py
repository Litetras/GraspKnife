import yaml
import os
import glob
import math
import numpy as np
import trimesh

import numpy as np
import trimesh

def find_true_boundary_by_thickness_profile(mesh_path, bins=80):
    """
    [基于顶视图厚度特征的算法]
    利用核心物理直觉：刀刃是一条长且极薄的线，而刀柄会显著变粗。
    通过统计沿长度（X轴）的厚度（Z轴）变化，寻找厚度突变（一阶导数最大）的点作为唯一分界。
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    extents = mesh.extents
    
    major_axis = np.argmax(extents)  # 长度方向 (通常是X)
    thin_axis = np.argmin(extents)   # 厚度方向 (通常是Z)
    # 因为完全不看高度了，所以不需要 medium_axis

    vertices = mesh.vertices
    v_major = vertices[:, major_axis]
    
    min_val, max_val = np.min(v_major), np.max(v_major)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    thicknesses = np.zeros(bins)
    bin_centers = np.zeros(bins)

    # 1. 提取每个切片的“绝对厚度”
    for i in range(bins):
        mask = (v_major >= bin_edges[i]) & (v_major <= bin_edges[i+1])
        pts = vertices[mask]
        
        bin_centers[i] = (bin_edges[i] + bin_edges[i+1]) / 2.0
        
        # 只计算厚度方向的极差
        if len(pts) > 0: 
            thicknesses[i] = np.max(pts[:, thin_axis]) - np.min(pts[:, thin_axis])
        else:
            thicknesses[i] = np.nan
            
    # 插值填补少数切片空洞
    def fill_nans(arr):
        mask = np.isnan(arr)
        if not np.all(mask) and np.any(mask):
            arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        elif np.all(mask):
            arr[:] = 0.0
        return arr
        
    thicknesses = fill_nans(thicknesses)
    
    # 2. 轻微平滑厚度曲线，抹平模型表面微小噪点，但不破坏阶跃特征
    # 使用稍微大一点的 kernel (5) 确保刀刃部分的细微厚度变化被无视
    kernel = np.ones(5) / 5.0
    thickness_smooth = np.convolve(thicknesses, kernel, mode='same')
    
    # 3. 🌟 核心：计算厚度突变率 🌟
    # 刀刃是平缓的细线（diff近乎0），刀柄变粗的瞬间（护手处）会产生一个巨大的峰值
    diff_thickness = np.abs(np.diff(thickness_smooth))
    
    if len(diff_thickness) == 0 or np.max(diff_thickness) == 0:
        return major_axis, np.mean(v_major), True
        
    # 4. 锁定“由细变粗”或“由粗变细”的瞬间
    max_diff_idx = np.argmax(diff_thickness)
    true_boundary_coord = bin_centers[max_diff_idx]
    
    # 5. 校验刀柄方向：边界两侧，平均厚度更大的一侧，绝对是刀柄
    thick_positive = np.mean(thicknesses[max_diff_idx+1:])
    thick_negative = np.mean(thicknesses[:max_diff_idx+1])
    
    handle_is_positive = thick_positive > thick_negative 
    
    return major_axis, true_boundary_coord, handle_is_positive
def process_and_split_grasps(yaml_path, mesh_dir, out_dirs, z_threshold, y_threshold, x_limit, max_tilt_angle):
    filename = os.path.basename(yaml_path)
    base_name = filename.replace('.yaml', '')
    
    # 寻找对应的 3D 模型文件 (兼容 .obj 和 .glb)
    mesh_path = os.path.join(mesh_dir, f"{base_name}.obj")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(mesh_dir, f"{base_name}.glb")
        if not os.path.exists(mesh_path):
            print(f"[{base_name}] Skipped: 3D Mesh not found.")
            return

    # --- 执行一维截面轮廓分析 ---2
    try:
        major_axis, boundary_coord, handle_is_positive = find_true_boundary_by_thickness_profile(mesh_path)
    except Exception as e:
        print(f"[{base_name}] Skipped: Trimesh parsing error - {e}")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    original_count = len(data.get('grasps', {}))
    if original_count == 0:
        return

    # 准备四个容器存放分类后的抓取
    categorized_grasps = {
        'up_handle': {}, 'up_blade': {},
        'down_handle': {}, 'down_blade': {}
    }

    for grasp_id, grasp_info in data['grasps'].items():
        pos = grasp_info['position']
        x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]
        
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # --- 原有的空间位置过滤 ---
        is_not_side = abs(z_pos) < z_threshold
        is_not_front_back_y = abs(y_pos) > y_threshold
        is_not_front_back_x = abs(x_pos) < x_limit

        # --- 原有的接近方向过滤 ---
        v_y = 2 * (qy * qz - qx * w)
        v_y_clamped = max(-1.0, min(1.0, v_y)) 
        angle_with_vertical = math.degrees(math.acos(abs(v_y_clamped)))
        is_vertical = angle_with_vertical < max_tilt_angle

        # 满足基础要求的优质抓取才进行语义分类
        if is_not_side and is_not_front_back_y and is_not_front_back_x and is_vertical:
            
            # 1. 判断全局几何方向 (Up / Down)
            direction = "up" if y_pos > 0.0 else "down"
            
            # 2. 判断功能部件 (Handle / Blade)
            grasp_major_val = pos[major_axis]
            if handle_is_positive:
                # 刀柄在正半区，坐标大于边界即为刀柄
                is_handle = grasp_major_val > boundary_coord
            else:
                # 刀柄在负半区，坐标小于等于边界即为刀柄
                is_handle = grasp_major_val <= boundary_coord
                
            part = "handle" if is_handle else "blade"
            
            # 3. 组合 Key 并存入
            key = f"{direction}_{part}"
            categorized_grasps[key][grasp_id] = grasp_info

    # --- 写入四个独立文件 ---
    kept_total = 0
    for key, grasps in categorized_grasps.items():
        if grasps:
            out_path = os.path.join(out_dirs[key], f"{base_name}_{key}.yaml")
            new_data = dict(data)  # 保留原有 header 等信息
            new_data['grasps'] = grasps
            with open(out_path, 'w') as f:
                yaml.dump(new_data, f, sort_keys=False)
            kept_total += len(grasps)

    removed_total = original_count - kept_total
    print(f"[{base_name}] Extracted: "
          f"U_Hand={len(categorized_grasps['up_handle']):<4} | "
          f"U_Blad={len(categorized_grasps['up_blade']):<4} | "
          f"D_Hand={len(categorized_grasps['down_handle']):<4} | "
          f"D_Blad={len(categorized_grasps['down_blade']):<4} "
          f"(Filtered {removed_total})")


def process_all(input_dir, mesh_dir, output_root, **kwargs):
    # 创建 4 个语义分类输出文件夹
    categories = ['up_handle', 'up_blade', 'down_handle', 'down_blade']
    out_dirs = {cat: os.path.join(output_root, cat) for cat in categories}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"Found {len(yaml_files)} YAML files.")
    print(f"Starting 1D Profile Analysis & Semantic Separation...\n" + "="*60)
    
    for yaml_file in yaml_files:
        process_and_split_grasps(
            yaml_path=yaml_file, 
            mesh_dir=mesh_dir, 
            out_dirs=out_dirs,
            **kwargs
        )
        
    print("="*60 + f"\n[Done] All grasps structurally grounded and separated!")
    print(f"Output saved to: {output_root}")


if __name__ == "__main__":
    # ------------------ 配置区 ------------------
    # 1. 原始全局抓取 YAML 文件目录
    INPUT_DIR = "/home/zyp/GraspDataGen/grasp_guess_data/franka_panda/" 
    
    # 2. 对应的 3D Mesh 目录 (.obj 或 .glb)
    #    非常重要：算法需要读取模型实体来计算厚度边界
    MESH_DIR = "/home/zyp/Desktop/knives_cleaned_aligned" 
    
    # 3. 最终输出的根目录（会自动创建 4 个子文件夹）
    OUTPUT_ROOT = "/home/zyp/Desktop/knives_grounded_grasps/" 
    
    # 执行批处理
    process_all(
        input_dir=INPUT_DIR, 
        mesh_dir=MESH_DIR, 
        output_root=OUTPUT_ROOT,
        z_threshold=0.02, 
        y_threshold=0.02, 
        x_limit=0.15,
        max_tilt_angle=10.0  # 沿用你收紧后的 10度 限制
    )
import os
import glob
import numpy as np
import trimesh

def align_mug_perfectly(input_path, output_path):
    # 1. 读取并合并网格
    scene_or_mesh = trimesh.load(input_path, force='mesh')
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return
        mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh

    # ==============================================================
    # 步骤 1：用 OBB 卡直模型，消除任何倾斜角
    # ==============================================================
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    mesh.apply_transform(to_origin)
    mesh.apply_translation(-mesh.centroid) 

    # ==============================================================
    # 步骤 2：找实心杯底，并强行让杯子立起来 (杯口朝 +Z，杯底朝 -Z)
    # ==============================================================
    vertices = mesh.vertices
    densities = []
    for axis in range(3):
        coords = vertices[:, axis]
        c_min, c_max = np.min(coords), np.max(coords)
        extent = c_max - c_min
        slice_thick = extent * 0.10  # 两端各切下 10% 的薄片

        count_min = np.sum(coords < c_min + slice_thick)
        count_max = np.sum(coords > c_max - slice_thick)

        if count_min > count_max:
            densities.append((count_min, -1, axis))
        else:
            densities.append((count_max, 1, axis))

    # 找出点数绝对最多的一面，必定是实心杯底
    best_bottom = max(densities, key=lambda x: x[0])
    z_sign = best_bottom[1]      
    height_axis = best_bottom[2] 

    # 重组坐标系，把杯底旋转到 -Z，杯口朝 +Z
    V_z = np.zeros(3)
    V_z[height_axis] = -z_sign  
    
    V_x = np.zeros(3)
    V_x[(height_axis + 1) % 3] = 1.0  # 暂时随便给一个正交轴
    V_y = np.cross(V_z, V_x)

    R_upright = np.eye(4)
    R_upright[0, :3] = V_x
    R_upright[1, :3] = V_y
    R_upright[2, :3] = V_z
    mesh.apply_transform(R_upright)

    # 确保杯子居中（绕着原点转）
    mesh.apply_translation(-mesh.centroid)

    # ==============================================================
    # 步骤 3：“雷达扫描”锁定手柄，旋转至 +X 轴
    # ==============================================================
    # 此时杯子已经笔直立在桌面上（Z轴），我们在水平面（X-Y平面）寻找把手
    xy_coords = mesh.vertices[:, :2]
    
    # 计算每个点到中心轴的距离平方
    dists_sq = np.sum(xy_coords**2, axis=1)
    
    # 提取距离最远的 2% 的点（防噪点），它们绝对是把手最外侧的那块肉
    top_k = max(5, int(len(dists_sq) * 0.02))
    furthest_indices = np.argsort(dists_sq)[-top_k:]
    furthest_points = xy_coords[furthest_indices]
    
    # 计算把手群的平均方向中心
    handle_vec = np.mean(furthest_points, axis=0)
    
    # 计算把手当前所在的角度
    angle = np.arctan2(handle_vec[1], handle_vec[0])
    
    # 绕着 Z 轴反向旋转这个角度，把手柄精准掰回 0 度（即 +X 轴正方向）
    R_z = trimesh.transformations.rotation_matrix(-angle, [0, 0, 1])
    mesh.apply_transform(R_z)

    # 最终严格居中
    mesh.apply_translation(-mesh.centroid)

    # ==============================================================
    # 导出纯净无杂质的 OBJ
    # ==============================================================
    mesh.export(output_path)
    
    mtl_path = output_path.replace(".obj", ".mtl")
    if os.path.exists(mtl_path):
        os.remove(mtl_path)
        
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(output_path, "w", encoding="utf-8") as f:
        clean_name = os.path.splitext(os.path.basename(output_path))[0]
        for line in lines:
            if not line.startswith("mtllib") and not line.startswith("usemtl"):
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {clean_name}\n")
                else:
                    f.write(line)

# ==========================================
# 批量执行逻辑
# ==========================================
if __name__ == "__main__":
    input_dir = "/home/zyp/pan1/objaverse_dataset_5/mugs_cleaned"
    output_dir = "/home/zyp/pan1/objaverse_dataset_5/mugs_cleaned_aligned"
    
    os.makedirs(output_dir, exist_ok=True)
    obj_files = glob.glob(os.path.join(input_dir, "*.obj"))
    
    print(f"\n🚀 开始完美对齐马克杯 (共 {len(obj_files)} 个)")
    
    success_count = 0
    for obj_path in obj_files:
        try:
            file_name = os.path.splitext(os.path.basename(obj_path))[0]
            output_path = os.path.join(output_dir, f"{file_name}_aligned.obj")
            
            align_mug_perfectly(obj_path, output_path)
            success_count += 1
            if success_count % 10 == 0:
                print(f"  -> 已成功对齐 {success_count} 个模型...")
        except Exception as e:
            print(f"  [失败] 处理 {os.path.basename(obj_path)} 时出错: {e}")

    print(f"\n🎉 马克杯完美对齐完成！成功数: {success_count}/{len(obj_files)}")
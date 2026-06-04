import os
import glob
import numpy as np
import open3d as o3d

def align_and_save_object(input_path, output_path, category):
    """
    通用 PCA 对齐 + 针对特定类别的启发式消歧 + 纯净导出
    """
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    # --- 阶段一：PCA 基础对齐 (通用) ---
    center = mesh.get_center()
    mesh.translate(-center)

    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    points = np.asarray(pcd.points)
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    R = eigenvectors.T
    mesh.rotate(R, center=(0, 0, 0))

    # 重新采样以进行启发式判断
    points = np.asarray(mesh.sample_points_uniformly(10000).points)

    if category == "mugs":
        # 现在的 points 已经经过了第一阶段 PCA 对齐。
        # === 按照“杯底实心，杯口空心”的物理特征强制对齐 ===

        # 1. 寻找杯底（扫描 6 个极端的切片，点最多的必是实心杯底）
        extremes = []
        for axis in range(3):
            coords = points[:, axis]
            c_min, c_max = np.min(coords), np.max(coords)
            extent = c_max - c_min
            slice_thick = extent * 0.10  # 切下 10% 厚度的薄片
            
            # 统计两端薄片内的表面点云数量
            count_min = np.sum(coords < c_min + slice_thick)
            count_max = np.sum(coords > c_max - slice_thick)
            
            extremes.append((count_min, axis, -1)) # -1 代表负方向极点
            extremes.append((count_max, axis, 1))  # 1 代表正方向极点
            
        # 找出点数最多的那个极端面（就是实心大圆盘杯底！）
        max_count, bottom_axis, bottom_sign = max(extremes, key=lambda item: item[0])
        
        # 2. 强制把杯底翻转到 -Z 轴
        if bottom_axis == 0: # 杯底在 X 轴
            if bottom_sign == 1:
                mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0)), center=(0,0,0))
            else:
                mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, -np.pi/2, 0)), center=(0,0,0))
        elif bottom_axis == 1: # 杯底在 Y 轴
            if bottom_sign == 1:
                mesh.rotate(mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0)), center=(0,0,0))
            else:
                mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=(0,0,0))
        elif bottom_axis == 2: # 杯底已经在 Z 轴
            if bottom_sign == 1: # 但它朝上(+Z)，需要翻面
                mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0,0,0))
                
        # 3. 寻找把手并对齐到 +X 轴
        # 重新采样确保最新坐标
        points = np.asarray(mesh.sample_points_uniformly(10000).points)
        
        # 在 X-Y 平面（水平面）寻找距离中心点最远的点
        dists_sq = points[:, 0]**2 + points[:, 1]**2
        handle_idx = np.argmax(dists_sq)
        
        # 计算把手所在的偏转角
        angle = np.arctan2(points[handle_idx, 1], points[handle_idx, 0])
        
        # 沿着 Z 轴反向旋转这个角度，把手就被死死锁在 +X 轴上了
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, -angle)), center=(0,0,0))

    elif category == "spoons":
        # 1. 区分勺把与勺头 (X轴)：勺头通常更宽 (Y的绝对值大)
        x_pos_width = np.mean(np.abs(points[points[:, 0] > 0][:, 1]))
        x_neg_width = np.mean(np.abs(points[points[:, 0] < 0][:, 1]))
        if x_pos_width > x_neg_width:
            # 如果 +X 更宽，说明勺头在右边，翻转180度让勺把朝 +X，勺头朝 -X
            mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi, 0)), center=(0,0,0))
            points = np.asarray(mesh.sample_points_uniformly(10000).points) # 更新点云

        # 2. 勺头凹面朝上 (Z轴)：提取勺头部分(-X)，评估Z轴分布
        bowl_pts = points[points[:, 0] < 0]
        if len(bowl_pts) > 0:
            z_mean = np.mean(bowl_pts[:, 2])
            z_mid = (np.max(bowl_pts[:, 2]) + np.min(bowl_pts[:, 2])) / 2.0
            # 凹面朝上时，底部点多，边缘点少且高，导致均值偏低
            if z_mean > z_mid: 
                mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0,0,0))

    elif category == "screwdrivers":
        # 1. 区分刀把与金属杆 (X轴)：刀把更粗 (Y和Z的分布更广)
        x_pos_thick = np.mean(np.abs(points[points[:, 0] > 0][:, 1]) + np.abs(points[points[:, 0] > 0][:, 2]))
        x_neg_thick = np.mean(np.abs(points[points[:, 0] < 0][:, 1]) + np.abs(points[points[:, 0] < 0][:, 2]))
        if x_neg_thick > x_pos_thick:
            # 让更粗的刀把朝向 +X
            mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi, 0)), center=(0,0,0))

    elif category == "electric_drills":
        # 1. 区分钻头与电机后部 (X轴)：电机后部比钻头更厚实 (Z轴厚度)
        x_pos_thick = np.mean(np.abs(points[points[:, 0] > 0][:, 2]))
        x_neg_thick = np.mean(np.abs(points[points[:, 0] < 0][:, 2]))
        if x_pos_thick > x_neg_thick:
            # 电机应朝向 -X，钻头朝向 +X
            mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi, 0)), center=(0,0,0))
            points = np.asarray(mesh.sample_points_uniformly(10000).points)

        # 2. 区分手柄上下 (Y轴)：电钻手柄向下延伸，所以 -Y 的长度应大于 +Y
        y_max = np.max(points[:, 1])
        y_min = np.min(points[:, 1])
        if abs(y_max) > abs(y_min):
            # 如果上部突出更多，说明拿反了，绕X轴翻面
            mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0,0,0))

    # --- 阶段三：纯净导出 ---
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, mesh, write_triangle_uvs=False, write_vertex_colors=False)

    mtl_path = output_path.replace(".obj", ".mtl")
    if os.path.exists(mtl_path):
        os.remove(mtl_path)
        
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.startswith("mtllib") and not line.startswith("usemtl"):
                f.write(line)

def batch_process():
    # 定义类别及其输入输出路径
    base_dir = "/home/zyp/pan1/objaverse_dataset_5"
    categories = {
        # "electric_drills": "electric_drills_cleaned",
        "mugs": "mugs_cleaned"
        # "screwdrivers": "screwdrivers_cleaned",
        # "spoons": "spoons_cleaned"
    }

    for cat_name, folder_name in categories.items():
        input_dir = os.path.join(base_dir, folder_name)
        output_dir = os.path.join(base_dir, folder_name + "_aligned")
        
        os.makedirs(output_dir, exist_ok=True)
        obj_files = glob.glob(os.path.join(input_dir, "*.obj"))
        obj_files = [f for f in obj_files if "_aligned.obj" not in f]

        if not obj_files:
            print(f"[{cat_name}] 在 {input_dir} 中未找到待处理文件，跳过。")
            continue

        print(f"\n开始处理类别: {cat_name.upper()} (找到 {len(obj_files)} 个模型)")
        print(f"输出目录: {output_dir}")
        
        success_count = 0
        for obj_path in obj_files:
            try:
                file_name = os.path.splitext(os.path.basename(obj_path))[0]
                output_path = os.path.join(output_dir, f"{file_name}_aligned.obj")
                
                align_and_save_object(obj_path, output_path, category=cat_name)
                success_count += 1
                if success_count % 10 == 0:
                    print(f"  -> 已成功对齐 {success_count} 个模型...")
            except Exception as e:
                print(f"  [失败] 处理 {os.path.basename(obj_path)} 时出错: {e}")

        print(f"[{cat_name}] 处理完成！成功数: {success_count}/{len(obj_files)}")

if __name__ == "__main__":
    batch_process()
    print("\n所有类别对齐完毕！")
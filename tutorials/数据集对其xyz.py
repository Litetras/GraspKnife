import os
import glob
import numpy as np
import open3d as o3d

def align_and_save_knife(input_path, output_path):
    """
    1. PCA 对齐轴向
    2. 启发式消歧统一正负朝向 (刀把+X, 刀背+Y)
    3. 输出纯净无材质 .obj
    """
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    # --- 第一阶段：PCA 基础对齐 ---
    center = mesh.get_center()
    mesh.translate(-center)

    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    points = np.asarray(pcd.points)
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # 防止镜像翻转
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    R = eigenvectors.T
    mesh.rotate(R, center=(0, 0, 0))

    # --- 第二阶段：X 轴消歧 (统一前后：刀把朝 +X) ---
    # 逻辑：刀把通常比刀刃部分更厚实、点云更多。
    aligned_points = np.asarray(mesh.sample_points_uniformly(10000).points)
    x_pos_count = np.sum(aligned_points[:, 0] > 0)
    x_neg_count = np.sum(aligned_points[:, 0] < 0)

    if x_neg_count > x_pos_count:
        # 如果 -X 方向的点更多，说明刀把朝向了左边。绕 Y 轴转 180 度调头。
        R_180_y = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
        mesh.rotate(R_180_y, center=(0, 0, 0))

    # --- 第三阶段：Y 轴消歧 (统一下上：刀背朝 +Y，刀刃朝 -Y) ---
    # 逻辑：刀背通常较厚，刀刃极薄。我们比较 Y>0 和 Y<0 区域的平均 Z 轴厚度。
    # 必须重新采样，因为前面可能发生过旋转
    aligned_points = np.asarray(mesh.sample_points_uniformly(10000).points)
    y_pos_points = aligned_points[aligned_points[:, 1] > 0]
    y_neg_points = aligned_points[aligned_points[:, 1] < 0]

    if len(y_pos_points) > 0 and len(y_neg_points) > 0:
        # 计算上下两半部分的平均厚度（绝对值）
        y_pos_thickness = np.mean(np.abs(y_pos_points[:, 2]))
        y_neg_thickness = np.mean(np.abs(y_neg_points[:, 2]))

        if y_neg_thickness > y_pos_thickness:
            # 如果下半部分比上半部分厚，说明刀背在下面。绕 X 轴转 180 度翻面。
            R_180_x = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            mesh.rotate(R_180_x, center=(0, 0, 0))

    # --- 第四阶段：纯净导出 ---
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

def batch_align_knives(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    obj_files = glob.glob(os.path.join(input_dir, "*.obj"))
    obj_files = [f for f in obj_files if "_aligned.obj" not in f]

    if not obj_files:
        print(f"在 {input_dir} 中没有找到需要处理的 .obj 文件。")
        return

    print(f"找到 {len(obj_files)} 个原始模型。")
    print(f"正在强制统一朝向并保存纯净 obj 至: {output_dir}")

    success_count = 0
    for obj_path in obj_files:
        try:
            file_name = os.path.splitext(os.path.basename(obj_path))[0]
            output_path = os.path.join(output_dir, f"{file_name}_aligned.obj")
            
            align_and_save_knife(obj_path, output_path)
            print(f"[成功] 绝对对齐并保存: {os.path.basename(output_path)}")
            success_count += 1
        except Exception as e:
            print(f"[失败] 处理 {os.path.basename(obj_path)} 时出错: {e}")

    print("-" * 30)
    print(f"批处理完成！成功统一处理 {success_count}/{len(obj_files)} 个模型。")

if __name__ == "__main__":
    INPUT_DIR = "/home/zyp/Desktop/my_kitchen_knives"
    OUTPUT_DIR = "/home/zyp/Desktop/my_kitchen_knives_aligned"
    batch_align_knives(INPUT_DIR, OUTPUT_DIR)
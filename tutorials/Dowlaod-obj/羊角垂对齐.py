import os
import glob
import numpy as np
import open3d as o3d

def align_and_save_hammer(input_path, output_path):
    """
    1. PCA 对齐轴向
    2. 启发式消歧统一正负朝向 (锤把朝 +X, 敲击面朝 +Y, 羊角朝 -Y)
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

    # --- 第二阶段：X 轴消歧 (统一前后：锤把朝 +X，锤头在 -X) ---
    # 逻辑：锤头所在的区域，其 Y 轴的跨度（长度）远大于锤把。
    aligned_points = np.asarray(mesh.sample_points_uniformly(10000).points)
    x_pos_points = aligned_points[aligned_points[:, 0] > 0]
    x_neg_points = aligned_points[aligned_points[:, 0] < 0]

    # 计算正负 X 区域的 Y 轴跨度
    y_spread_pos = np.max(x_pos_points[:, 1]) - np.min(x_pos_points[:, 1]) if len(x_pos_points) > 0 else 0
    y_spread_neg = np.max(x_neg_points[:, 1]) - np.min(x_neg_points[:, 1]) if len(x_neg_points) > 0 else 0

    if y_spread_pos > y_spread_neg:
        # 如果 +X 方向的 Y 跨度更大，说明锤头在右边。绕 Y 轴转 180 度，调头。
        R_180_y = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
        mesh.rotate(R_180_y, center=(0, 0, 0))

    # --- 第三阶段：Y 轴消歧 (统一下上：敲击面朝 +Y，羊角朝 -Y) ---
    # 逻辑：此时锤头在 -X 区域。敲击面是实心的，厚度（Z轴）大；羊角是分叉渐薄的，厚度小。
    aligned_points = np.asarray(mesh.sample_points_uniformly(10000).points)
    # 只取锤头部分的点云进行判断 (X < 0)
    head_points = aligned_points[aligned_points[:, 0] < 0] 
    
    y_pos_head = head_points[head_points[:, 1] > 0]
    y_neg_head = head_points[head_points[:, 1] < 0]

    if len(y_pos_head) > 0 and len(y_neg_head) > 0:
        # 计算上下两半部分的平均厚度（Z的绝对值）
        y_pos_thickness = np.mean(np.abs(y_pos_head[:, 2]))
        y_neg_thickness = np.mean(np.abs(y_neg_head[:, 2]))

        if y_neg_thickness > y_pos_thickness:
            # 如果下半部分比上半部分厚，说明实心的敲击面在下面。绕 X 轴转 180 度翻面。
            R_180_x = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            mesh.rotate(R_180_x, center=(0, 0, 0))

    # --- 第四阶段：纯净导出 ---
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, mesh, write_triangle_uvs=False, write_vertex_colors=False)

    # 移除残留的 mtl 引用
    mtl_path = output_path.replace(".obj", ".mtl")
    if os.path.exists(mtl_path):
        os.remove(mtl_path)
        
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.startswith("mtllib") and not line.startswith("usemtl"):
                f.write(line)

def batch_align_hammers(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    obj_files = glob.glob(os.path.join(input_dir, "*.obj"))
    obj_files = [f for f in obj_files if "_aligned.obj" not in f]

    if not obj_files:
        print(f"在 {input_dir} 中没有找到需要处理的 .obj 文件。")
        return

    print(f"找到 {len(obj_files)} 个羊角锤模型。")
    print(f"正在强制统一朝向并保存纯净 obj 至: {output_dir}")

    success_count = 0
    for obj_path in obj_files:
        try:
            file_name = os.path.splitext(os.path.basename(obj_path))[0]
            output_path = os.path.join(output_dir, f"{file_name}.obj")
            
            align_and_save_hammer(obj_path, output_path)
            print(f"[成功] 对齐并保存: {os.path.basename(output_path)}")
            success_count += 1
        except Exception as e:
            print(f"[失败] 处理 {os.path.basename(obj_path)} 时出错: {e}")

    print("-" * 30)
    print(f"批处理完成！成功统一处理 {success_count}/{len(obj_files)} 个模型。")

if __name__ == "__main__":
    # 请修改为你的羊角锤路径，例如之前挑选出来的 final_selection
    INPUT_DIR = "/home/zyp/Desktop/hammers"
    OUTPUT_DIR = "/home/zyp/Desktop/hammers_cleaned_aligned"
    batch_align_hammers(INPUT_DIR, OUTPUT_DIR)
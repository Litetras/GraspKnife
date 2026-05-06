import trimesh
import numpy as np
import os

# ================= 配置区 =================
WORK_DIR = "/home/zyp/Desktop/Brush/"
REF_FILENAME = "brush_1.obj"  # 指定用于作为基准的文件
OUTPUT_DIR = os.path.join(WORK_DIR, "Aligned")
TARGET_SIZE = 0.3  # 统一缩放尺寸 (30cm)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ==========================================

def get_rotation_matrix(axis, theta):
    """生成绕指定轴旋转theta弧度的旋转矩阵"""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c), 0],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b), 0],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c, 0],
        [0, 0, 0, 1]
    ])

def normalize_and_pca_align(mesh):
    """将模型居中、对齐主轴并缩放"""
    # 1. 移动到几何重心
    mesh.vertices -= mesh.center_mass
    
    # 2. 惯性主轴对齐 (PCA核心操作，将几何分布方向对齐到 XYZ)
    transform = mesh.principal_inertia_transform
    mesh.apply_transform(transform)
    
    # 3. 统一缩放到目标尺寸
    max_side = np.max(mesh.extents)
    if max_side > 0:
        mesh.apply_scale(TARGET_SIZE / max_side)
    
    # 重新确保在中心
    mesh.vertices -= mesh.center_mass
    return mesh

def main():
    ref_path = os.path.join(WORK_DIR, REF_FILENAME)
    if not os.path.exists(ref_path):
        print(f"错误: 找不到基准文件 {ref_path}")
        return

    print(f"--- 步骤 1: 自动对齐基准模型 ({REF_FILENAME}) ---")
    ref_mesh = trimesh.load(ref_path, force='mesh')
    
    # 对基准模型进行纯几何的 PCA 对齐
    ref_mesh = normalize_and_pca_align(ref_mesh)
    
    # 导出自动对齐后的基准模型
    ref_out_path = os.path.join(OUTPUT_DIR, f"auto_aligned_{REF_FILENAME}")
    ref_mesh.export(ref_out_path)
    print(f"基准模型已自动对齐并保存至: {ref_out_path}\n")

    # 定义 4 种可能的 180 度翻转初始态 
    flips = [
        np.eye(4), # 不翻转
        get_rotation_matrix(np.array([1, 0, 0]), np.pi), # 绕X转180
        get_rotation_matrix(np.array([0, 1, 0]), np.pi), # 绕Y转180
        get_rotation_matrix(np.array([0, 0, 1]), np.pi)  # 绕Z转180
    ]

    print("--- 步骤 2: 处理其他模型并强制匹配基准 ---")
    for filename in os.listdir(WORK_DIR):
        if filename.endswith(".obj") and filename != REF_FILENAME:
            filepath = os.path.join(WORK_DIR, filename)
            print(f"正在处理: {filename} ...")
            
            try:
                # 强制作为单网格加载
                mesh = trimesh.load(filepath, force='mesh') 
                
                # A. 先让它自己通过 PCA 对齐，做到形状大体归正
                mesh = normalize_and_pca_align(mesh)
                
                # B. 穷举 ICP 匹配，寻找最佳姿态
                best_cost = float('inf')
                best_matrix = np.eye(4)
                
                for flip_mat in flips:
                    temp_mesh = mesh.copy()
                    temp_mesh.apply_transform(flip_mat)
                    
                    # [修复] 修改了参数并接收正确的2个返回值
                    matrix, cost = trimesh.registration.mesh_other(
                        temp_mesh, 
                        ref_mesh, 
                        samples=500,  
                        scale=False,  
                        icp_first=10,
                        icp_final=50
                    )
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_matrix = np.dot(matrix, flip_mat)
                
                # 应用计算出的最佳变换矩阵
                mesh.apply_transform(best_matrix)
                
                # 导出最终结果
                out_path = os.path.join(OUTPUT_DIR, f"aligned_{filename}")
                mesh.export(out_path)
                print(f"  -> 成功 (误差: {best_cost:.4f})")
                
            except Exception as e:
                print(f"  -> 处理失败: {e}")

    print("\n全部处理完成！请查看 Aligned 文件夹。")

if __name__ == "__main__":
    main()
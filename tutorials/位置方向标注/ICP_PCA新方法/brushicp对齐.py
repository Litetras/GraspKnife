import trimesh
import numpy as np
import os

# ================= 配置区 =================
WORK_DIR = "/home/zyp/Desktop/Brush/"
REF_FILENAME = "brush_1.obj"  # 手动标注好、作为绝对基准的文件
OUTPUT_DIR = os.path.join(WORK_DIR, "Aligned")

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

def main():
    ref_path = os.path.join(WORK_DIR, REF_FILENAME)
    if not os.path.exists(ref_path):
        print(f"错误: 找不到基准文件 {ref_path}")
        return

    print(f"--- 步骤 1: 加载手动标注的基准模型 ({REF_FILENAME}) ---")
    # 【关键修改 1】直接加载，不再进行PCA对齐，保持你手动标注的原始姿态和位置！
    ref_mesh = trimesh.load(ref_path, force='mesh')
    
    # 提取基准模型的中心点和真实尺寸，留作后用
    ref_center = ref_mesh.center_mass
    ref_size = np.max(ref_mesh.extents)
    print(f"基准模型加载成功！中心点: {ref_center.round(3)}, 最大尺寸: {ref_size:.3f}")

    # 定义 4 种可能的 180 度翻转初始态 
    flips = [
        np.eye(4), # 不翻转
        get_rotation_matrix(np.array([1, 0, 0]), np.pi), # 绕X转180
        get_rotation_matrix(np.array([0, 1, 0]), np.pi), # 绕Y转180
        get_rotation_matrix(np.array([0, 0, 1]), np.pi)  # 绕Z转180
    ]

    print("\n--- 步骤 2: 处理其他模型并向基准模型靠拢 ---")
    for filename in os.listdir(WORK_DIR):
        if filename.endswith(".obj") and filename != REF_FILENAME:
            filepath = os.path.join(WORK_DIR, filename)
            print(f"正在处理: {filename} ...")
            
            try:
                mesh = trimesh.load(filepath, force='mesh') 
                
                # A. 基础预处理：在原点处整理好待处理模型的姿态和大小
                mesh.vertices -= mesh.center_mass
                mesh.apply_transform(mesh.principal_inertia_transform)
                max_side = np.max(mesh.extents)
                if max_side > 0:
                    # 【关键修改 2】严格按照 brush1 的真实尺寸进行缩放
                    mesh.apply_scale(ref_size / max_side)
                mesh.vertices -= mesh.center_mass # 确保整理完依然在原点
                
                # B. 穷举姿态并进行 ICP 匹配
                best_cost = float('inf')
                best_final_mesh = None
                
                for flip_mat in flips:
                    temp_mesh = mesh.copy()
                    
                    # 1. 先在原点处进行测试翻转（如果在别的地方翻转会导致模型飞走）
                    temp_mesh.apply_transform(flip_mat)
                    
                    # 2. 【关键修改 3】粗对齐：把翻转后的模型平移到 brush1 所在的空间位置
                    # 这一步非常重要！否则距离太远 ICP 会直接匹配失败
                    temp_mesh.vertices += ref_center
                    
                    # 3. 运行 ICP 微调
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
                        # 将跑完 ICP 的最终姿态保存下来
                        best_final_mesh = temp_mesh.copy()
                        best_final_mesh.apply_transform(matrix)
                
                # 导出最终结果
                if best_final_mesh is not None:
                    out_path = os.path.join(OUTPUT_DIR, f"aligned_{filename}")
                    best_final_mesh.export(out_path)
                    print(f"  -> 成功 (误差: {best_cost:.4f})")
                
            except Exception as e:
                print(f"  -> 处理失败: {e}")

    print("\n全部处理完成！请查看 Aligned 文件夹。")

if __name__ == "__main__":
    main()
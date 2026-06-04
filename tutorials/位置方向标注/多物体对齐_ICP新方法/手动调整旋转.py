import os
import open3d as o3d
import numpy as np

# ==========================================
# 配置区域
# ==========================================
WORK_DIR = "/home/zyp/pan1/objaverse_dataset_5/mugs_cleaned_aligned"  # 你存放对齐后模型的文件夹

# 【修复清单】
# 格式: {"文件名": "旋转操作"}
# 旋转操作代号：
# 'rx180' : 绕 X 轴转 180 度 (解决：上下反了，羊角朝上了)
# 'ry180' : 绕 Y 轴转 180 度 (解决：左右反了，锤头跑到右边了)
# 'rz90'  : 绕 Z 轴逆时针转 90 度 (解决：锤把竖起来了，跑到 Y 轴上了)
# 'rz-90' : 绕 Z 轴顺时针转 90 度 (同上)

fixes = {
    "mug_0e9285f3_aligned.obj": "rx180",  # 例子：把15号的上下翻转一下
    "mug_2b3e80b2_aligned.obj": "rx180",   # 例子：把29号躺平
    "mug_2bb1d7dd_aligned.obj": "rx180",  
    "mug_2be0f0d2_aligned.obj": "rx180", 
    "mug_2f3c24f3_aligned.obj": "rx180",
    "mug_3b943184_aligned.obj": "rx180" ,
     "mug_3d42c50b_aligned.obj": "rx180",
     "mug_3e1b1c9e_aligned.obj": "rx180",
     "mug_3ef58711_aligned.obj": "rx180",
     "mug_3f3eb0ed_aligned.obj": "rx180",
     "mug_4cc68f98_aligned.obj": "rx180",
     "mug_4f04c5c4_aligned.obj": "rx180",
     "mug_5a68b7ad_aligned.obj": "rx180"


}

# ==========================================
# 执行修复
# ==========================================
def fix_orientation():
    for filename, action in fixes.items():
        filepath = os.path.join(WORK_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[警告] 找不到文件: {filename}")
            continue

        mesh = o3d.io.read_triangle_mesh(filepath)
        center = mesh.get_center() # 绕自身中心旋转

        if action == 'rx180':
            R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        elif action == 'ry180':
            R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
        elif action == 'rz90':
            R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
        elif action == 'rz-90':
            R = mesh.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))
        else:
            print(f"未知的操作代号: {action}")
            continue

        mesh.rotate(R, center=(0, 0, 0))
        
        # 重新计算法线并覆盖保存
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filepath, mesh, write_triangle_uvs=False, write_vertex_colors=False)
        print(f"[修复成功] {filename} 执行了 {action}")

if __name__ == "__main__":
    fix_orientation()
    print("指定模型的微调已完成！")
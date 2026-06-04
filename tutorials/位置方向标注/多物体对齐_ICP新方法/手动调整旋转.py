import os
import shutil
import time

import numpy as np
import open3d as o3d

# ==========================================
# 配置区域
# ==========================================
# 这个脚本只做“手动指定文件 -> 手动指定旋转”的修补。
# 适合 ICP/PCA 把 key、fork 这类细长物体语义方向对歪之后，逐个纠正。
#
# 旋转操作可以写字符串，也可以写列表，例如：
#   "fork_3.obj": "rz90"
#   "key_7.obj": ["rx180", "rz-90"]
#
# 支持的操作代号：
#   rx90 / rx-90 / rx180
#   ry90 / ry-90 / ry180
#   rz90 / rz-90 / rz180
# ==========================================

FIX_JOBS = {
    "forks": {
        "work_dir": "/home/zyp/Desktop/objaverse_dataset/forks",
        "fixes": {
            # 示例：

            "fork_12.obj": "rx180",

            # "fork_5.obj": ["rx180", "rz-90"],
        },
    },
    "keys": {
        "work_dir": "/home/zyp/Desktop/objaverse_dataset/keys",
        "fixes": {
            # 示例：
            # "key_3.obj": "rx180",
            # "key_8.obj": ["ry180", "rz90"],
        },
    },
}

BACKUP_BEFORE_WRITE = True
BACKUP_DIR_NAME_PREFIX = "_manual_rotation_backup"


def get_rotation_matrix(mesh, action):
    if action == "rx90":
        return mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    if action == "rx-90":
        return mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    if action == "rx180":
        return mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))

    if action == "ry90":
        return mesh.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    if action == "ry-90":
        return mesh.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
    if action == "ry180":
        return mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))

    if action == "rz90":
        return mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
    if action == "rz-90":
        return mesh.get_rotation_matrix_from_xyz((0, 0, -np.pi / 2))
    if action == "rz180":
        return mesh.get_rotation_matrix_from_xyz((0, 0, np.pi))

    raise ValueError(f"未知旋转操作: {action}")


def normalize_actions(actions):
    if isinstance(actions, str):
        return [actions]
    if isinstance(actions, (list, tuple)):
        return list(actions)
    raise TypeError(f"旋转操作必须是字符串或列表，收到: {type(actions)}")


def backup_file(filepath, backup_root):
    os.makedirs(backup_root, exist_ok=True)
    backup_path = os.path.join(backup_root, os.path.basename(filepath))
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
    return backup_path


def apply_rotation_fix(work_dir, filename, actions, backup_root):
    filepath = os.path.join(work_dir, filename)
    if not os.path.exists(filepath):
        print(f"[警告] 找不到文件: {filepath}")
        return False

    if BACKUP_BEFORE_WRITE:
        backup_path = backup_file(filepath, backup_root)
        print(f"  🛟 备份: {backup_path}")

    mesh = o3d.io.read_triangle_mesh(filepath)
    if mesh.is_empty():
        print(f"[警告] 空模型，跳过: {filepath}")
        return False

    # 这里沿用你原脚本的行为：绕世界原点旋转。
    # 如果模型已经整体居中，这比绕自身几何中心更容易保持全类物体坐标系一致。
    for action in normalize_actions(actions):
        rotation = get_rotation_matrix(mesh, action)
        mesh.rotate(rotation, center=(0, 0, 0))

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(
        filepath,
        mesh,
        write_triangle_uvs=False,
        write_vertex_colors=False,
    )
    print(f"[修复成功] {filename} 执行 {normalize_actions(actions)}")
    return True


def fix_category(category_name, config):
    work_dir = config["work_dir"]
    fixes = config.get("fixes", {})

    print("\n" + "=" * 70)
    print(f"开始手动旋转修补类别: {category_name}")
    print(f"目录: {work_dir}")
    print(f"待修补文件数: {len(fixes)}")

    if not os.path.exists(work_dir):
        print(f"❌ 找不到目录: {work_dir}")
        return 0

    if not fixes:
        print("⚠️ fixes 为空，没有执行任何旋转。请先在配置里填写要修补的文件。")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(work_dir, f"{BACKUP_DIR_NAME_PREFIX}_{timestamp}")

    success_count = 0
    for filename, actions in fixes.items():
        print("-" * 60)
        print(f"处理: {filename} -> {actions}")
        try:
            if apply_rotation_fix(work_dir, filename, actions, backup_root):
                success_count += 1
        except Exception as exc:
            print(f"[失败] {filename}: {exc}")

    print(f"✅ {category_name} 完成: {success_count}/{len(fixes)}")
    return success_count


def main():
    total_success = 0
    for category_name, config in FIX_JOBS.items():
        total_success += fix_category(category_name, config)

    print("\n" + "=" * 70)
    print(f"指定模型的手动旋转微调完成，总计修补成功: {total_success} 个")
    print("=" * 70)


if __name__ == "__main__":
    main()

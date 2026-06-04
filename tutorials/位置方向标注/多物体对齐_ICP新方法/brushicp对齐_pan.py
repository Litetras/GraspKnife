import trimesh
import numpy as np
import os
import itertools
import shutil
import time
import json

# ================= 配置区 =================
SOURCE_DIR = "/home/zyp/Desktop/objaverse_dataset/pans"

# 作为绝对基准的 Pan
REF_FILENAME = "pan_6c8fb584.obj"

# 是否在最后替换原始 pans 文件夹
REPLACE_ORIGINAL = True

# 是否保留临时对齐文件夹
KEEP_TEMP_ALIGNED_DIR = False
# ==========================================


def as_mesh(obj):
    """
    兼容 trimesh.Scene / trimesh.Trimesh。
    如果是 Scene，则合并所有 geometry。
    """
    if isinstance(obj, trimesh.Scene):
        if len(obj.geometry) == 0:
            return None
        return trimesh.util.concatenate(tuple(obj.geometry.values()))
    return obj


def make_transform_from_rotation(R):
    """
    3x3 rotation -> 4x4 transform
    """
    T = np.eye(4)
    T[:3, :3] = R
    return T


def generate_24_axis_rotations():
    """
    生成 24 种合法坐标轴旋转。
    比只试 X/Y/Z 的 180 度翻转更稳，可以处理 90 度轴错位。
    """
    rotations = []

    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3))

            for row, col in enumerate(perm):
                R[row, col] = signs[row]

            # 只保留 det = +1 的旋转，排除镜像反射
            if np.isclose(np.linalg.det(R), 1.0):
                rotations.append(R)

    return rotations


def preprocess_mesh_to_origin(mesh, ref_size):
    """
    对待对齐模型做基础预处理：
    1. 移到原点
    2. PCA 主惯性轴粗对齐
    3. 缩放到基准 pan 的尺寸
    4. 再次居中
    """
    mesh = mesh.copy()

    # 移到原点
    mesh.apply_translation(-mesh.center_mass)

    # PCA / 主惯性轴粗对齐
    try:
        mesh.apply_transform(mesh.principal_inertia_transform)
    except Exception as e:
        print(f"  -> ⚠️ PCA 对齐失败，仅使用原始姿态。原因: {e}")

    # 再次居中
    mesh.apply_translation(-mesh.center_mass)

    # 缩放到基准模型尺寸
    max_side = np.max(mesh.extents)
    if max_side > 0:
        mesh.apply_scale(ref_size / max_side)

    # 最后再居中
    mesh.apply_translation(-mesh.center_mass)

    return mesh


def safe_backup_source_dir(source_dir):
    """
    备份整个 pans 文件夹。
    """
    parent_dir = os.path.dirname(source_dir)
    folder_name = os.path.basename(source_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(
        parent_dir,
        f"{folder_name}_backup_before_align_{timestamp}"
    )

    print("\n--- 步骤 0: 备份原始 pans 文件夹 ---")
    print(f"📦 原始目录: {source_dir}")
    print(f"🛟 备份目录: {backup_dir}")

    shutil.copytree(source_dir, backup_dir)

    print("✅ 备份完成！")
    return backup_dir


def replace_original_folder(source_dir, temp_aligned_dir):
    """
    用对齐后的文件夹替换原始 pans 文件夹。
    """
    print("\n--- 步骤 3: 替换原始 pans 文件夹 ---")
    print(f"🗑️ 即将删除原始目录: {source_dir}")
    print(f"📥 使用对齐目录替换: {temp_aligned_dir}")

    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)

    shutil.copytree(temp_aligned_dir, source_dir)

    print("✅ 已把对齐后的模型放回原始路径！")
    print(f"📁 当前可继续使用路径: {source_dir}")


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 找不到源目录: {SOURCE_DIR}")
        return

    ref_path_check = os.path.join(SOURCE_DIR, REF_FILENAME)
    if not os.path.exists(ref_path_check):
        print(f"❌ 找不到基准模型: {ref_path_check}")
        return

    print("=" * 70)
    print("🍳 启动 Pan 专属模型对齐 + 原路径覆盖脚本")
    print(f"📌 基准模型: {REF_FILENAME}")
    print(f"📁 原始 Pan 目录: {SOURCE_DIR}")
    print("=" * 70)

    # 0. 先完整备份原始 pans 文件夹
    backup_dir = safe_backup_source_dir(SOURCE_DIR)

    parent_dir = os.path.dirname(SOURCE_DIR)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_aligned_dir = os.path.join(parent_dir, f"pans_aligned_tmp_{timestamp}")
    os.makedirs(temp_aligned_dir, exist_ok=True)

    print("\n--- 步骤 1: 加载基准 Pan 模型 ---")

    # 注意：从备份目录读取，避免后面替换时影响源文件
    ref_path = os.path.join(backup_dir, REF_FILENAME)

    ref_obj = trimesh.load(ref_path, force="mesh")
    ref_mesh = as_mesh(ref_obj)

    if ref_mesh is None:
        print("❌ 基准模型加载失败。")
        return

    # 基准模型不做 PCA，不改姿态，保持它作为绝对标准
    ref_center = ref_mesh.center_mass
    ref_size = np.max(ref_mesh.extents)

    print("✅ 基准模型加载成功")
    print(f"   中心点: {ref_center.round(4)}")
    print(f"   最大尺寸: {ref_size:.6f}")

    # 把基准模型也导出到临时对齐目录
    ref_out_path = os.path.join(temp_aligned_dir, REF_FILENAME)
    ref_mesh.export(ref_out_path)
    print(f"✅ 基准模型已写入临时对齐目录: {ref_out_path}")

    # 生成 24 种候选旋转
    candidate_rotations = generate_24_axis_rotations()
    print(f"✅ 已生成 {len(candidate_rotations)} 种候选轴向旋转。")

    print("\n--- 步骤 2: 对齐其它 Pan 模型 ---")

    mesh_files = [
        f for f in os.listdir(backup_dir)
        if f.lower().endswith((".obj", ".glb")) and f != REF_FILENAME
    ]

    mesh_files = sorted(mesh_files)

    print(f"📦 待处理模型数量: {len(mesh_files)}")

    success_count = 0
    fail_count = 0
    transform_log = {}

    for idx, filename in enumerate(mesh_files, start=1):
        filepath = os.path.join(backup_dir, filename)

        print("\n" + "-" * 60)
        print(f"正在处理 [{idx}/{len(mesh_files)}]: {filename}")

        try:
            obj = trimesh.load(filepath, force="mesh")
            mesh = as_mesh(obj)

            if mesh is None:
                print("  -> ❌ 空模型，跳过。")
                fail_count += 1
                continue

            # A. 预处理到原点、PCA、缩放到基准尺寸
            mesh = preprocess_mesh_to_origin(mesh, ref_size)

            # B. 穷举 24 种轴向旋转 + ICP
            best_cost = float("inf")
            best_final_mesh = None
            best_rot_idx = -1
            best_matrix = None

            for rot_idx, R in enumerate(candidate_rotations):
                temp_mesh = mesh.copy()

                # 1. 在原点处应用候选旋转
                T_rot = make_transform_from_rotation(R)
                temp_mesh.apply_transform(T_rot)

                # 2. 粗平移到基准模型中心附近
                temp_mesh.apply_translation(ref_center)

                # 3. ICP 微调
                try:
                    matrix, cost = trimesh.registration.mesh_other(
                        temp_mesh,
                        ref_mesh,
                        samples=800,
                        scale=False,
                        icp_first=10,
                        icp_final=60
                    )
                except Exception:
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_rot_idx = rot_idx
                    best_matrix = matrix

                    best_final_mesh = temp_mesh.copy()
                    best_final_mesh.apply_transform(matrix)

            # C. 导出结果
            if best_final_mesh is not None:
                stem = os.path.splitext(filename)[0]

                # 统一输出 obj，保持 base_name 不变
                out_filename = f"{stem}.obj"
                out_path = os.path.join(temp_aligned_dir, out_filename)

                best_final_mesh.export(out_path)

                transform_log[stem] = {
                    "source_filename": filename,
                    "output_filename": out_filename,
                    "best_cost": float(best_cost),
                    "best_rotation_index": int(best_rot_idx),
                    "icp_matrix": best_matrix.tolist() if best_matrix is not None else None
                }

                print(f"  -> ✅ 成功对齐: {out_filename}")
                print(f"     最佳误差: {best_cost:.6f}")
                print(f"     最佳旋转编号: {best_rot_idx}")

                success_count += 1

            else:
                print("  -> ❌ ICP 未找到有效结果，跳过。")
                fail_count += 1

        except Exception as e:
            print(f"  -> ❌ 处理失败: {e}")
            fail_count += 1

    # 保存对齐日志
    log_path = os.path.join(temp_aligned_dir, "pan_alignment_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(transform_log, f, indent=4)

    print("\n--- 对齐统计 ---")
    print(f"✅ 成功对齐: {success_count} 个")
    print(f"❌ 失败跳过: {fail_count} 个")
    print(f"📝 对齐日志: {log_path}")
    print(f"📁 临时对齐目录: {temp_aligned_dir}")

    # 4. 替换原始目录
    if REPLACE_ORIGINAL:
        replace_original_folder(SOURCE_DIR, temp_aligned_dir)

        if not KEEP_TEMP_ALIGNED_DIR:
            print("\n--- 步骤 4: 清理临时对齐目录 ---")
            shutil.rmtree(temp_aligned_dir)
            print(f"✅ 已删除临时目录: {temp_aligned_dir}")

    print("\n" + "=" * 70)
    print("🎉 Pan 对齐完成！")
    print(f"🛟 原始模型备份在: {backup_dir}")
    print(f"📁 当前模型路径仍然是: {SOURCE_DIR}")
    print("=" * 70)

    print("\n⚠️ 接下来建议重新生成这些 JSON：")
    print("   1. pan_dataset_boundaries_auto.json")
    print("   2. pan_category_grasp_directions.json")
    print("   3. final_pan_task_oriented_dataset.json")
    print("因为旧 JSON 是基于未对齐模型生成的。")


if __name__ == "__main__":
    main()
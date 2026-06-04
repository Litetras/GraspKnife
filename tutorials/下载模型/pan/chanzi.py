import os
import re
import json
import time
import random
import trimesh
import objaverse

# ==========================================
# 代理配置区域
# ==========================================
PROXY_URL = "http://127.0.0.1:7897"

os.environ["ALL_PROXY"] = PROXY_URL
os.environ["all_proxy"] = PROXY_URL
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL

print(f"已设置网络代理为: {PROXY_URL}")

# ==========================================
# 全局配置
# ==========================================
BASE_OUTPUT_DIR = "/home/zyp/Desktop/objaverse_dataset"
MAX_RETRIES = 5

# 只想先筛 UID 不下载，就改成 True
ONLY_EXTRACT_UIDS = False

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()


# ==========================================
# 文本匹配工具函数
# ==========================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_keyword(text, keyword):
    keyword = normalize_text(keyword)
    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def has_any_keyword(text, keywords):
    return any(has_keyword(text, k) for k in keywords)


# ==========================================
# 锅铲 / 煎铲筛选规则
# ==========================================

categories = {
    "spatula": {
        "name_zh": "锅铲 / 煎铲 / 炒菜铲 (Spatula / Turner)",

        # 强核心词：明确是厨房锅铲
        "strong_core_keywords": {
            "spatula",
            "kitchen spatula",
            "cooking spatula",
            "frying spatula",
            "turner",
            "kitchen turner",
            "pancake turner",
            "fish turner",
            "slotted spatula",
            "silicone spatula",
            "wooden spatula",
            "metal spatula",
            "stainless steel spatula",
            "wok spatula",
            "egg spatula",
            "food flipper",
            "burger flipper",
            "grill spatula",
            "bbq spatula"
        },

        # 弱核心词：需要配合厨房语境
        "weak_core_keywords": {
            "turner",
            "flipper"
        },

        # 厨房上下文
        "context_keywords": {
            "kitchen",
            "cooking",
            "cook",
            "food",
            "frying",
            "fry",
            "pan",
            "wok",
            "egg",
            "pancake",
            "fish",
            "burger",
            "bbq",
            "grill",
            "chef",
            "utensil",
            "tableware",
            "cookware",
            "handle",
            "silicone",
            "wooden",
            "metal",
            "stainless"
        },

        # 排除非厨房 spatula / 非锅铲
        "exclude_keywords": {
            "medical",
            "dental",
            "lab",
            "laboratory",
            "chemistry",
            "paint",
            "painting",
            "palette knife",
            "putty knife",
            "scraper",
            "wall scraper",
            "cake server",
            "icing spatula",
            "makeup",
            "cosmetic",

            "weapon",
            "character",
            "monster",
            "robot",
            "logo",
            "icon",
            "sign"
        },

        "uid_file": "spatula_uids.json",
        "output_dir": os.path.join(BASE_OUTPUT_DIR, "spatulas"),
        "target_scale": 0.22,
        "prefix": "spatula"
    }
}


# ==========================================
# 第一阶段：筛选 UID
# ==========================================

final_uids = {cat: [] for cat in categories}

print("\n正在搜集锅铲 / 煎铲 / 炒菜铲模型数据...")

for uid, item in annotations.items():
    name = str(item.get("name", ""))
    description = str(item.get("description", ""))

    tag_strings = []
    for tag in item.get("tags", []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get("name", "")))
        else:
            tag_strings.append(str(tag))

    tags_text = " ".join(tag_strings)

    raw_full_text = f"{name} {description} {tags_text}"
    full_text = normalize_text(raw_full_text)

    for cat_key, rules in categories.items():
        has_strong_core = has_any_keyword(full_text, rules["strong_core_keywords"])
        has_weak_core = has_any_keyword(full_text, rules["weak_core_keywords"])
        has_context = has_any_keyword(full_text, rules["context_keywords"])
        has_excluded = has_any_keyword(full_text, rules["exclude_keywords"])

        # 判断逻辑：
        # 1. 出现 spatula / kitchen spatula / wok spatula 等强核心词
        # 2. 出现 turner / flipper，同时出现 kitchen / cooking / pan / food 等上下文
        # 3. 排除 medical / paint / scraper / icing spatula 等非锅铲
        is_likely_spatula = (
            has_strong_core or
            (has_weak_core and has_context)
        )

        if is_likely_spatula and not has_excluded:
            final_uids[cat_key].append(uid)

# ==========================================
# 结果输出与保存 UID
# ==========================================

print("\n" + "=" * 60)
print("筛选完成！提取统计：")
for cat_key, rules in categories.items():
    print(f"  - {rules['name_zh']} 模型总数: {len(final_uids[cat_key])}")
print("=" * 60)

for cat_key, rules in categories.items():
    output_file = rules["uid_file"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_uids[cat_key], f, ensure_ascii=False, indent=2)

    print(f"\n{rules['name_zh']} UID列表已保存至: {output_file}")

    if final_uids[cat_key]:
        print(f"前 20 个 {rules['name_zh']} 模型预览:")
        for i in range(min(20, len(final_uids[cat_key]))):
            uid = final_uids[cat_key][i]
            data = annotations[uid]
            print(f"  [{i + 1}] 名称: {data.get('name')} | UID: {uid[:12]}...")

if ONLY_EXTRACT_UIDS:
    print("\n已设置 ONLY_EXTRACT_UIDS=True，只生成 UID，不下载模型。")
    exit()


# ==========================================
# 第二阶段：下载 + 转换 OBJ
# ==========================================

print("\n" + "=" * 60)
print("开始下载并转换锅铲模型 OBJ ...")
print("=" * 60)

for category, config in categories.items():
    print("\n" + "=" * 60)
    print(f"🚀 开始处理类别: {category.upper()}")
    print("=" * 60)

    uid_file = config["uid_file"]
    out_dir = config["output_dir"]
    target_scale = config["target_scale"]
    prefix = config["prefix"]

    if not os.path.exists(uid_file):
        print(f"⚠️ 找不到 UID 文件: {uid_file}，跳过该类别！")
        continue

    with open(uid_file, "r", encoding="utf-8") as f:
        uids = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[{category}] 已加载 {len(uids)} 个模型 UID，输出目录: {out_dir}")

    success_count = 0
    skip_count = 0
    fail_download_count = 0
    fail_convert_count = 0

    for i, uid in enumerate(uids):
        obj_filename = f"{prefix}_{uid[:8]}.obj"
        obj_out_path = os.path.join(out_dir, obj_filename)

        if os.path.exists(obj_out_path):
            skip_count += 1
            if skip_count % 10 == 0:
                print(f"  -> ⏭️ 已跳过 {skip_count} 个已存在的模型...")
            continue

        download_success = False
        glb_path = None

        for attempt in range(MAX_RETRIES):
            try:
                objects = objaverse.load_objects(
                    uids=[uid],
                    download_processes=1
                )

                if uid in objects:
                    glb_path = objects[uid]
                    download_success = True
                    break

            except Exception as e:
                print(
                    f"  -> ❌ 模型 [{i + 1}/{len(uids)}] 下载失败 "
                    f"(尝试 {attempt + 1}/{MAX_RETRIES})"
                )

                if attempt < MAX_RETRIES - 1:
                    time.sleep(3)
                else:
                    print(f"  -> ⚠️ 放弃当前模型 UID: {uid[:8]}")

        if not download_success or glb_path is None:
            fail_download_count += 1
            continue

        try:
            scene_or_mesh = trimesh.load(glb_path, force="mesh")

            if isinstance(scene_or_mesh, trimesh.Scene):
                if len(scene_or_mesh.geometry) == 0:
                    fail_convert_count += 1
                    continue

                mesh = trimesh.util.concatenate(
                    tuple(scene_or_mesh.geometry.values())
                )
            else:
                mesh = scene_or_mesh

            if mesh is None or mesh.is_empty:
                fail_convert_count += 1
                continue

            mesh.apply_translation(-mesh.centroid)

            max_length = mesh.extents.max()
            if max_length > 0:
                scale_factor = target_scale / max_length
                mesh.apply_scale(scale_factor)

            mesh.export(obj_out_path)

            with open(obj_out_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            with open(obj_out_path, "w", encoding="utf-8") as f:
                clean_name = obj_filename.replace(".obj", "")
                wrote_name = False

                for line in lines:
                    if line.startswith("o ") or line.startswith("g "):
                        if not wrote_name:
                            f.write(f"o {clean_name}\n")
                            wrote_name = True
                    else:
                        f.write(line)

            success_count += 1
            print(f"  -> ✅ 成功 [{i + 1}/{len(uids)}]: {obj_filename}")

            time.sleep(random.uniform(0.1, 0.5))

        except Exception as e:
            fail_convert_count += 1
            print(f"  -> ⚠️ 模型 [{i + 1}/{len(uids)}] 转换时报错，已跳过。")
            continue

    print("\n" + "-" * 60)
    print(f"[{category}] 处理完毕！")
    print(f"  ✅ 新增生成: {success_count}")
    print(f"  ⏭️ 已存在跳过: {skip_count}")
    print(f"  ❌ 下载失败: {fail_download_count}")
    print(f"  ⚠️ 转换失败: {fail_convert_count}")

print("\n🎉 锅铲 / 煎铲 / 炒菜铲 的 UID 获取、下载与 OBJ 转换已全部完成！")
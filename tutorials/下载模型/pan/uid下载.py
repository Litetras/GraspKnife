import objaverse
import json
import re

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 文本匹配工具函数
# ==========================================

def normalize_text(text):
    """
    统一文本格式：
    - 小写
    - 将下划线、连字符、斜杠替换为空格
    - 合并多余空格
    """
    text = text.lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_keyword(text, keyword):
    """
    词边界匹配，避免 saw 匹配到 sawmill，key 匹配到 keyboard 的一部分。
    支持多词短语，比如 hand saw / door key / usb flash drive。
    """
    keyword = normalize_text(keyword)
    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def has_any_keyword(text, keywords):
    return any(has_keyword(text, k) for k in keywords)


# ==========================================
# 三类别筛选规则
# ==========================================

categories = {
    "hand_saw": {
        "name_zh": "手锯 / 手动锯 (Hand Saw)",

        # 强核心词：出现这些基本就是手动锯
        "strong_core_keywords": {
            "hand saw",
            "handsaw",
            "manual saw",

            # 木工手锯
            "wood saw",
            "wooden saw",
            "woodworking saw",
            "carpenter saw",
            "carpentry saw",
            "panel saw",
            "rip saw",
            "crosscut saw",
            "tenon saw",
            "back saw",

            # 金属手锯
            "hacksaw",
            "hack saw",
            "metal saw",
            "junior hacksaw",

            # 其它手动锯
            "coping saw",
            "bow saw",
            "fret saw",
            "pruning saw",
            "garden saw"
        },

        # 弱核心词：只有 saw 时必须配合上下文
        "weak_core_keywords": {
            "saw"
        },

        # 上下文词：强调手持、锯条、木头/金属切割
        "context_keywords": {
            "hand",
            "manual",
            "tool",
            "handle",
            "blade",
            "teeth",
            "tooth",
            "cut",
            "cutting",
            "wood",
            "wooden",
            "woodwork",
            "woodworking",
            "carpenter",
            "carpentry",
            "metal",
            "iron",
            "steel",
            "workshop",
            "hardware",
            "diy",
            "repair",
            "garden",
            "pruning"
        },

        # 排除电锯、机器锯、工业场景、非目标物体
        "exclude_keywords": {
            "chainsaw",
            "chain saw",
            "circular saw",
            "table saw",
            "miter saw",
            "mitre saw",
            "band saw",
            "bandsaw",
            "jigsaw",
            "jig saw",
            "reciprocating saw",
            "sabre saw",
            "saber saw",
            "power saw",
            "electric saw",
            "machine saw",
            "saw machine",

            "sawmill",
            "saw mill",
            "sawhorse",
            "saw horse",
            "factory",
            "industrial machine",

            "saw blade only",
            "circular blade",
            "blade disk",
            "blade disc",

            "weapon",
            "sword",
            "axe",
            "knife",
            "character",
            "monster",
            "robot",
            "vehicle",
            "logo",
            "sign",
            "game"
        }
    },

    "key": {
        "name_zh": "钥匙 (Key)",

        # 强核心词：明确是实体钥匙
        "strong_core_keywords": {
            "key",
            "door key",
            "house key",
            "car key",
            "metal key",
            "old key",
            "antique key",
            "skeleton key",
            "padlock key",
            "lock key",
            "golden key",
            "silver key",
            "brass key",
            "keychain key",
            "key ring",
            "keyring"
        },

        # 弱核心词：key 单独出现容易误匹配键盘键，所以配合上下文
        "weak_core_keywords": {
            "key"
        },

        # 上下文词：偏向真实钥匙 / 门锁 / 金属小物体
        "context_keywords": {
            "lock",
            "door",
            "house",
            "car",
            "padlock",
            "metal",
            "brass",
            "silver",
            "gold",
            "ring",
            "keyring",
            "keychain",
            "unlock",
            "security",
            "handle",
            "teeth"
        },

        # 排除键盘键、琴键、软件 key、游戏道具等
        "exclude_keywords": {
            "keyboard",
            "keyboard key",
            "keycap",
            "key cap",
            "piano",
            "piano key",
            "organ key",
            "synth",
            "musical",
            "typewriter",
            "button",
            "hotkey",
            "shortcut",
            "license key",
            "product key",
            "api key",
            "steam key",
            "key card",
            "keycard",
            "key blade",
            "keyblade",
            "character",
            "monster",
            "weapon",
            "logo",
            "icon",
            "symbol",
            "map key",
            "legend"
        }
    },

    "usb_flash_drive": {
        "name_zh": "U盘 / USB闪存盘 (USB Flash Drive)",

        # 强核心词：明确是 U盘
        "strong_core_keywords": {
            "usb flash drive",
            "flash drive",
            "thumb drive",
            "usb stick",
            "memory stick",
            "pen drive",
            "pendrive",
            "jump drive",
            "usb memory",
            "usb drive",
            "flash disk",
            "u disk",
            "u-disk"
        },

        # 弱核心词：单独 usb / drive / memory 容易误匹配
        "weak_core_keywords": {
            "usb",
            "drive",
            "flash",
            "memory"
        },

        # 上下文词：强调小型便携存储设备
        "context_keywords": {
            "storage",
            "data",
            "portable",
            "memory",
            "flash",
            "usb",
            "stick",
            "drive",
            "connector",
            "metal",
            "plastic",
            "cap",
            "device",
            "electronics",
            "computer",
            "laptop"
        },

        # 排除非 U盘：线缆、接口、硬盘、鼠标键盘、充电器等
        "exclude_keywords": {
            "usb cable",
            "cable",
            "wire",
            "charger",
            "adapter",
            "power adapter",
            "usb charger",
            "hub",
            "usb hub",
            "keyboard",
            "mouse",
            "controller",
            "gamepad",
            "hard drive",
            "hdd",
            "ssd",
            "external hard drive",
            "portable hard drive",
            "disk drive",
            "cd drive",
            "dvd drive",
            "floppy drive",
            "card reader",
            "sd card",
            "micro sd",
            "phone",
            "camera",
            "robot",
            "vehicle",
            "character",
            "logo",
            "icon",
            "symbol"
        }
    }
}

# 初始化结果字典
final_uids = {cat: [] for cat in categories}

print("正在搜集模型数据：手锯 / 钥匙 / U盘 ...")

for uid, item in annotations.items():
    # 1. 文本提取与格式化
    name = str(item.get("name", ""))
    description = str(item.get("description", ""))

    # 兼容标签格式
    tag_strings = []
    for tag in item.get("tags", []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get("name", "")))
        else:
            tag_strings.append(str(tag))

    tags_text = " ".join(tag_strings)

    raw_full_text = f"{name} {description} {tags_text}"
    full_text = normalize_text(raw_full_text)

    # 2. 并行判断三个类别
    for cat_key, rules in categories.items():
        has_strong_core = has_any_keyword(full_text, rules["strong_core_keywords"])
        has_weak_core = has_any_keyword(full_text, rules["weak_core_keywords"])
        has_context = has_any_keyword(full_text, rules["context_keywords"])
        has_excluded = has_any_keyword(full_text, rules["exclude_keywords"])

        # ==========================================
        # 宽松但防误匹配的判断逻辑
        # ==========================================
        # 满足以下任意一种：
        # 1. 出现强核心词，例如 hand saw / door key / usb flash drive
        # 2. 出现弱核心词，同时出现上下文词
        #
        # 同时不能出现明显排异词
        # ==========================================
        is_likely_target = (
            has_strong_core or
            (has_weak_core and has_context)
        )

        if is_likely_target and not has_excluded:
            final_uids[cat_key].append(uid)

# ==========================================
# 结果输出与保存
# ==========================================

print("\n" + "=" * 60)
print("筛选完成！提取统计：")
for cat_key, rules in categories.items():
    print(f"  - {rules['name_zh']} 模型总数: {len(final_uids[cat_key])}")
print("=" * 60)

# 保存 UID 列表并打印预览
for cat_key, rules in categories.items():
    output_file = f"{cat_key}_uids.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_uids[cat_key], f, ensure_ascii=False, indent=2)

    print(f"\n{rules['name_zh']} UID列表已保存至: {output_file}")

    if final_uids[cat_key]:
        print(f"前 20 个 {rules['name_zh']} 模型预览:")
        for i in range(min(20, len(final_uids[cat_key]))):
            uid = final_uids[cat_key][i]
            data = annotations[uid]
            print(f"  [{i + 1}] 名称: {data.get('name')} | UID: {uid[:12]}...")
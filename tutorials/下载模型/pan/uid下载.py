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
    词边界匹配，避免 fork 匹配到 forklift，pan 匹配到 panel。
    也支持多词短语，比如 spray bottle / watering can。
    """
    keyword = normalize_text(keyword)
    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def has_any_keyword(text, keywords):
    return any(has_keyword(text, k) for k in keywords)


# ==========================================
# 多类别筛选规则
# ==========================================

categories = {
    "trowel": {
        "name_zh": "园艺小铲 / 泥铲 (Trowel)",
        "strong_core_keywords": {
            "trowel", "garden trowel", "hand trowel", "gardening trowel",
            "planting trowel", "transplanting trowel", "soil trowel",
            "small shovel", "garden shovel"
        },
        "weak_core_keywords": {
            "spade", "shovel"
        },
        "context_keywords": {
            "garden", "gardening", "soil", "plant", "planting",
            "dig", "digging", "yard", "tool", "handle", "metal",
            "mud", "flower", "farm"
        },
        "exclude_keywords": {
            "snow shovel", "excavator", "tractor", "vehicle", "weapon",
            "character", "monster", "toy character", "logo"
        }
    },

    "fork": {
        "name_zh": "叉子 (Fork)",
        "strong_core_keywords": {
            "dinner fork", "table fork", "kitchen fork", "eating fork",
            "cutlery fork", "fork utensil", "salad fork", "dessert fork"
        },
        "weak_core_keywords": {
            "fork"
        },
        "context_keywords": {
            "kitchen", "cutlery", "utensil", "tableware", "silverware",
            "food", "eat", "eating", "dining", "metal", "restaurant",
            "plate", "spoon", "knife"
        },
        "exclude_keywords": {
            "forklift", "bike", "bicycle", "motorcycle", "road",
            "tuning fork", "pitchfork", "garden fork", "replication fork",
            "git", "code", "tree", "branch", "weapon", "character",
            "vehicle"
        }
    },

    "spray_bottle": {
        "name_zh": "喷雾瓶 / 喷壶 (Spray Bottle)",
        "strong_core_keywords": {
            "spray bottle", "sprayer", "trigger sprayer", "trigger spray",
            "pump sprayer", "pump spray bottle", "mist bottle",
            "water sprayer"
        },
        "weak_core_keywords": {
            "spray", "nozzle", "trigger"
        },
        "context_keywords": {
            "bottle", "pump", "clean", "cleaning", "water",
            "garden", "mist", "liquid", "dispenser", "plastic",
            "handle", "sprayer"
        },
        "exclude_keywords": {
            "spray paint", "paint", "graffiti", "particle", "effect",
            "fx", "blood", "splash", "weapon", "gun", "flamethrower",
            "character", "logo"
        }
    },

    "watering_can": {
        "name_zh": "浇花壶 / 洒水壶 (Watering Can)",
        "strong_core_keywords": {
            "watering can", "wateringcan", "watering pot",
            "garden watering can", "metal watering can",
            "plastic watering can"
        },
        "weak_core_keywords": {
            "watering"
        },
        "context_keywords": {
            "can", "garden", "plant", "flower", "water",
            "spout", "handle", "gardening", "yard", "watering"
        },
        "exclude_keywords": {
            "trash can", "garbage can", "soda can", "tin can",
            "oil can", "gas can", "watering hole", "character",
            "vehicle", "logo"
        }
    },

    "toilet_brush": {
        "name_zh": "马桶刷 (Toilet Brush)",
        "strong_core_keywords": {
            "toilet brush", "wc brush", "bathroom toilet brush",
            "toilet cleaner brush"
        },
        "weak_core_keywords": {
            "brush"
        },
        "context_keywords": {
            "toilet", "wc", "bathroom", "restroom", "cleaning",
            "cleaner", "holder", "bath", "lavatory"
        },
        "exclude_keywords": {
            "toothbrush", "tooth brush", "hair brush", "paint brush",
            "makeup brush", "brushes", "broom", "character", "weapon",
            "logo"
        }
    },

    "squeegee": {
        "name_zh": "刮水板 / 玻璃刮 (Squeegee)",
        "strong_core_keywords": {
            "squeegee", "squeege", "window squeegee", "glass squeegee",
            "water squeegee", "shower squeegee", "floor squeegee",
            "rubber squeegee"
        },
        "weak_core_keywords": {
            "wiper", "scraper"
        },
        "context_keywords": {
            "window", "glass", "water", "clean", "cleaning",
            "rubber", "shower", "bathroom", "windshield", "floor",
            "handle"
        },
        "exclude_keywords": {
            "paint scraper", "ice scraper", "weapon", "character",
            "graffiti", "effect", "particle", "logo"
        }
    },

    "dustpan": {
        "name_zh": "簸箕 (Dustpan)",
        "strong_core_keywords": {
            "dustpan", "dust pan"
        },
        "weak_core_keywords": {
            "pan"
        },
        "context_keywords": {
            "broom", "sweep", "sweeping", "dust", "clean",
            "cleaning", "trash", "garbage", "floor", "household",
            "janitor", "handle"
        },
        "exclude_keywords": {
            "frying pan", "saucepan", "cooking pan", "skillet",
            "panda", "panther", "panel", "panorama", "weapon",
            "character", "vehicle", "logo"
        }
    },

    "back_scratcher": {
        "name_zh": "痒痒挠 (Back Scratcher)",
        "strong_core_keywords": {
            "back scratcher", "backscratcher", "back scratching tool",
            "itch scratcher"
        },
        "weak_core_keywords": {
            "scratcher"
        },
        "context_keywords": {
            "back", "itch", "scratch", "scratching", "massage",
            "bamboo", "hand", "claw", "handle", "tool"
        },
        "exclude_keywords": {
            "cat scratcher", "dog scratcher", "scratching post",
            "scratch pad", "scratchpad", "animal", "character",
            "monster", "weapon", "logo"
        }
    },

    "charger": {
        "name_zh": "手机充电头 (Charger)",
        "strong_core_keywords": {
            "phone charger", "mobile charger", "usb charger",
            "wall charger", "charging brick", "power adapter",
            "charger adapter", "battery charger", "fast charger"
        },
        "weak_core_keywords": {
            "charger", "adapter"
        },
        "context_keywords": {
            "phone", "mobile", "usb", "cable", "plug", "wall",
            "socket", "power", "battery", "charging", "electronics",
            "electric", "device", "wire"
        },
        "exclude_keywords": {
            "dodge charger", "car", "vehicle", "horse", "knight",
            "football", "team", "logo", "pokemon", "robot",
            "weapon", "character", "spaceship"
        }
    },

    "pizza_cutter": {
        "name_zh": "披萨滚刀 / 披萨刀 (Pizza Cutter)",
        "strong_core_keywords": {
            "pizza cutter", "pizza wheel", "pizza knife",
            "pizza slicer", "rotary pizza cutter", "pizza roller"
        },
        "weak_core_keywords": {
            "cutter", "wheel", "slicer", "knife"
        },
        "context_keywords": {
            "pizza", "kitchen", "food", "cut", "cutting",
            "utensil", "blade", "handle", "restaurant"
        },
        "exclude_keywords": {
            "bike wheel", "car wheel", "wheelchair", "vehicle",
            "weapon", "sword", "character", "logo"
        }
    },

    "peeler": {
        "name_zh": "削皮器 / 去皮刀 (Peeler)",
        "strong_core_keywords": {
            "peeler", "vegetable peeler", "potato peeler",
            "fruit peeler", "apple peeler", "kitchen peeler",
            "y peeler", "y shaped peeler"
        },
        "weak_core_keywords": {
            "peel"
        },
        "context_keywords": {
            "vegetable", "potato", "carrot", "apple", "fruit",
            "kitchen", "utensil", "tool", "blade", "handle",
            "cook", "cooking", "food", "metal"
        },
        "exclude_keywords": {
            "skin", "face", "body", "character", "monster",
            "paint peeler", "paint", "wall", "industrial",
            "machine", "banana peel", "orange peel", "logo"
        }
    },

    "tiller": {
        "name_zh": "园艺翻土器 / 耕耘机 (Tiller)",
        "strong_core_keywords": {
            "tiller", "garden tiller", "rototiller", "rotary tiller",
            "soil tiller", "power tiller", "cultivator",
            "garden cultivator"
        },
        "weak_core_keywords": {
            "cultivate", "cultivator"
        },
        "context_keywords": {
            "garden", "soil", "farm", "farming", "agriculture",
            "cultivate", "rotary", "tractor", "machine", "engine",
            "tool", "yard", "land", "digging"
        },
        "exclude_keywords": {
            "boat", "ship", "rudder", "steering", "sailing",
            "sailboat", "handlebar", "scooter", "military",
            "training", "music", "character", "weapon", "logo"
        }
    }
}

# 初始化结果字典
final_uids = {cat: [] for cat in categories}

print("正在搜集模型数据（多类别宽松筛选：强核心词直接命中，弱核心词需上下文）...")

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

    # 2. 并行判断各类别逻辑
    for cat_key, rules in categories.items():
        has_strong_core = has_any_keyword(full_text, rules["strong_core_keywords"])
        has_weak_core = has_any_keyword(full_text, rules["weak_core_keywords"])
        has_context = has_any_keyword(full_text, rules["context_keywords"])
        has_excluded = has_any_keyword(full_text, rules["exclude_keywords"])

        # ==========================================
        # 宽松但防误匹配的判断逻辑
        # ==========================================
        # 满足以下任意一种：
        # 1. 出现强核心词，比如 "pizza cutter", "watering can", "dustpan"
        # 2. 出现弱核心词，同时出现上下文词，比如 fork + kitchen / cutlery
        #
        # 同时不能出现排异词
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
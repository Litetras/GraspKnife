import objaverse
import json

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 核心过滤逻辑：多类别并行提取
# ==========================================

categories = {
    "mug": {
        "name_zh": "马克杯 (Mug)",
        "core_keyword": "mug",
        # 【收紧】去除了宽泛的 cup/drink/hot 等，只保留明确的“咖啡、茶、陶瓷、瓷器、搪瓷”
        "context_keywords": {"coffee", "tea", "ceramic", "porcelain", "enamel"},
        # 【新增排异】加入了 trophy(奖杯), tavern/medieval/fantasy(酒馆/中世纪/奇幻木杯), wood(木头), broken(破损)
        "exclude_keywords": {"smug", "shot", "face", "head", "beer", "magic", "character", "monster", "bottle", "trophy", "tavern", "medieval", "fantasy", "wood", "root", "broken"}
    },
    "spoon": {
        "name_zh": "勺子 (Spoon)",
        "core_keyword": "spoon",
        "context_keywords": {"soup", "tea", "coffee", "dessert", "eat", "kitchen", "cutlery", "utensil", "silverware", "food", "tableware", "metal", "wood"},
        "exclude_keywords": {"spoonbill", "bird", "fork", "knife", "spork", "ladle", "scoop", "shoe", "weapon", "character"}
    },
    "screwdriver": {
        "name_zh": "螺丝刀 (Screwdriver)",
        "core_keyword": "screwdriver",
        "context_keywords": {"tool", "philips", "flathead", "screw", "hardware", "repair", "fix", "mechanic", "work", "diy", "equipment", "handle"},
        "exclude_keywords": {"drink", "cocktail", "vodka", "juice", "orange", "sonic", "doctor", "who", "character", "robot"}
    },
    "electric_drill": {
        "name_zh": "电钻 (Electric Drill)",
        "core_keyword": "drill",
        "context_keywords": {"electric", "power", "cordless", "battery", "tool", "hardware", "hand", "chuck", "dewalt", "makita", "bosch", "diy"},
        "exclude_keywords": {"oil", "rig", "dentist", "dental", "press", "mine", "mining", "military", "training", "music", "earth", "character"}
    },
    "sprayer": {
        "name_zh": "喷雾器 (Sprayer)",
        "core_keyword": "spray",
        "context_keywords": {"bottle", "sprayer", "pump", "nozzle", "trigger", "clean", "garden", "water", "aerosol", "can", "dispenser"},
        "exclude_keywords": {"effect", "particle", "blood", "sea", "ocean", "graffiti", "fx", "splash", "weapon", "gun"}
    }
}

# 初始化结果字典
final_uids = {cat: [] for cat in categories}

print("正在搜集模型数据（合并遍历以提升效率，已应用严格的排异逻辑）...")

for uid, item in annotations.items():
    # 1. 文本提取与格式化
    name = str(item.get('name', '')).lower()
    description = str(item.get('description', '')).lower()
    
    # 兼容标签格式
    tag_strings = []
    for tag in item.get('tags', []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get('name', '')).lower())
        else:
            tag_strings.append(str(tag).lower())
    tags_text = ' '.join(tag_strings)
    
    full_text = f"{name} {description} {tags_text}"
    
    # 2. 并行判断各类别逻辑
    for cat_key, rules in categories.items():
        has_core = rules["core_keyword"] in full_text
        has_context = any(k in full_text for k in rules["context_keywords"])
        has_excluded = any(k in full_text for k in rules["exclude_keywords"])
        
        # 必须包含核心词 + 至少一个场景词 + 不能包含任何黑名单词
        if has_core and has_context and not has_excluded:
            final_uids[cat_key].append(uid)

# ==========================================
# 结果输出与保存
# ==========================================
print("\n" + "="*60)
print("筛选完成！提取统计：")
for cat_key, rules in categories.items():
    print(f"  - {rules['name_zh']} 模型总数: {len(final_uids[cat_key])}")
print("="*60)

# 保存各个UID列表并打印预览
for cat_key, rules in categories.items():
    output_file = f"{cat_key}_uids.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_uids[cat_key], f)
    
    print(f"\n{rules['name_zh']} UID列表已保存至: {output_file}")
    
    # 打印前 5 个模型预览
    if final_uids[cat_key]:
        print(f"前 5 个 {rules['name_zh']} 模型预览:")
        for i in range(min(5, len(final_uids[cat_key]))):
            uid = final_uids[cat_key][i]
            data = annotations[uid]
            print(f"  [{i+1}] 名称: {data.get('name')} | UID: {uid[:12]}...")
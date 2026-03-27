import objaverse
import json
import re  # 引入正则库进行全词匹配

print("正在加载 Objaverse XL 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 终极过滤逻辑（严防死守版）
# ==========================================

# 【第一层：必须是确切的刀】
# 删除了 blade 和 cutlery，新增了厨刀的特定称呼
must_have_keywords = [
    'knife', 'knives', 'cleaver', 'chopper', 'santoku', 'nakiri'
]

# 【第二层：全方位黑名单】
exclude_keywords = [
    # 1. 漏网的武器/战斗相关
    'sword', 'dagger', 'weapon', 'military', 'tactical', 'combat', 'army', 'navy',
    'katana', 'samurai', 'bayonet', 'bowie', 'machete', 'kunai', 'shuriken',
    'survival', 'hunting', 'ninja', 'karambit', 'blood', 'zombie', 'assassin',
    'battle', 'executioner', 'war', 'spear', 'warrior', 'killer',
    
    # 2. 折叠/便携刀具
    'pocket', 'folding', 'switchblade', 'butterfly', 'balisong', 'swiss', 'multi',
    
    # 3. 奇幻/游戏道具风
    'fantasy', 'medieval', 'sci-fi', 'scifi', 'cyberpunk', 'rpg', 'magic', 'dungeon',
    
    # 4. 其他带刃工具
    'axe', 'hatchet', 'scythe', 'sickle', 'scalpel', 'saw', 'chainsaw', 'gauge', 'tool',
    
    # 5. 【新增】绝对拒绝“套装”和“其他餐具”，防止混入包含刀的场景和碗碟
    'set', 'pack', 'collection', 'bundle', 'scene', 'table', 'setting',
    'bowl', 'plate', 'fork', 'spoon', 'cup', 'mug', 'pan', 'pot', 'dish'
]

# 将关键词编译为正则表达式，实现“全词匹配”（只匹配单独的单词，不匹配字母组合）
# 例如：\bknife\b 会匹配 "kitchen knife"，但不会匹配 "penknife"
must_regex = re.compile(r'\b(?:' + '|'.join(must_have_keywords) + r')\b')
exclude_regex = re.compile(r'\b(?:' + '|'.join(exclude_keywords) + r')\b')

print("正在执行严格清洗逻辑...")
final_uids = []

for uid, item in annotations.items():
    name = item.get('name', '').lower()
    description = item.get('description', '').lower()
    
    tag_strings = []
    for tag in item.get('tags', []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get('name', '')).lower())
        else:
            tag_strings.append(str(tag).lower())
    tags_text = ' '.join(tag_strings)
    
    # 核心策略：我们更信任名字和标签，描述里往往有很多废话蹭热度
    title_tags_text = f"{name} {tags_text}"
    full_text = f"{name} {tags_text} {description}"
    
    # 判断条件：
    # 1. 名字或标签里，必须明确包含刀具核心词汇（避免只是描述里提了一句）
    has_must = bool(must_regex.search(title_tags_text))
    
    # 2. 全文（包含描述）绝对不能包含任何黑名单词汇
    has_exclude = bool(exclude_regex.search(full_text))
    
    if has_must and not has_exclude:
        final_uids.append(uid)

# ==========================================
print("="*60)
print(f"筛选完成！经过严格清洗，纯净刀具数量: {len(final_uids)}")
print("="*60)

output_file = "kitchen_knife_pure_uids.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_uids, f)

if final_uids:
    print("\n前10个纯净模型预览:")
    for i in range(min(10, len(final_uids))):
        uid = final_uids[i]
        data = annotations[uid]
        print(f"  [{i+1}] 名称: {data.get('name')}")
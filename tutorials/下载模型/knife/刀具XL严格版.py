import objaverse
import json

print("正在加载 Objaverse XL 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 三层刚性过滤逻辑（从粗到精，精准锁定厨房刀具）
# ==========================================

# 【第一层：核心必选】没有这些词的直接pass（先锁死「刀具」这个大前提，彻底排除非刀具模型）
must_have_keywords = {'knife', 'knives', 'blade' , 'knives', 'cleaver', 
    'dagger', 'machete', 'scalpel', 'karambit', 'kunai'}

# 【第二层：厨房白名单】必须包含至少一个，确保是「厨房用」，而非其他用途的刀
kitchen_keywords = {
    'kitchen', 'chef', 'cooking', 'cleaver', 
    'fruit knife', 'paring knife', 'bread knife',
    'vegetable knife', 'carving knife', 'cutlery'
}

# 【第三层：武器黑名单】包含任意一个直接pass，彻底排除军刀/武器/冷兵器
exclude_keywords = {
    'sword', 'dagger', 'weapon', 'military', 'tactical',
    'katana', 'samurai', 'survival', 'hunting', 'bayonet',
    'switchblade', 'butterfly', 'karambit', 'ninja',
    'fantasy', 'medieval', 'army', 'gun', 'rifle', 'combat'
}

print("正在精准筛选【厨房专用刀具】...")
final_uids = []

for uid, item in annotations.items():
    # 1. 文本提取与格式化
    name = item.get('name', '').lower()
    description = item.get('description', '').lower()
    
    # 兼容标签格式
    tag_strings = []
    for tag in item.get('tags', []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get('name', '')).lower())
        else:
            tag_strings.append(str(tag).lower())
    tags_text = ' '.join(tag_strings)
    
    full_text = f"{name} {description} {tags_text}"
    
    # 2. 三层过滤判断（必须同时满足）
    # 条件1：必须包含刀具核心词
    has_must = any(k in full_text for k in must_have_keywords)
    # 条件2：必须包含厨房相关词
    has_kitchen = any(k in full_text for k in kitchen_keywords)
    # 条件3：绝对不能包含武器黑名单词
    has_exclude = any(k in full_text for k in exclude_keywords)
    
    if has_must and has_kitchen and not has_exclude:
        final_uids.append(uid)

# ==========================================
# 结果输出与保存
# ==========================================
print("="*60)
print(f"筛选完成！")
print(f"  - 原始泛刀具总数: ~2515")
print(f"  - 精准厨房刀具数量: {len(final_uids)}")
print("="*60)

# 保存UID列表
output_file = "kitchen_knife_uids_precise.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_uids, f)
print(f"\n精准UID列表已保存至: {output_file}")

# 打印前10个模型预览，确保全是厨房刀具
if final_uids:
    print("\n前10个厨房刀具模型预览:")
    for i in range(min(10, len(final_uids))):
        uid = final_uids[i]
        data = annotations[uid]
        print(f"  [{i+1}] 名称: {data.get('name')} | UID: {uid[:12]}...")

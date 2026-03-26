import objaverse
import json

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 核心过滤逻辑：只要刀，绝对不要剑
# ==========================================

# 【想要的核心词】包含各种单手可以握持的刀/刃
must_have_keywords = {
    'knife', 'knives', 'blade', 'cleaver', 
    'dagger', 'machete', 'scalpel', 'karambit', 'kunai'
}

# 【拒绝的黑名单词】将所有长剑、双手武器和光剑拉黑
exclude_keywords = {
    'sword', 'katana', 'rapier', 'claymore', 'saber', 
    'scimitar', 'broadsword', 'longsword', 'greatsword', 'lightsaber'
}

print("正在搜集广泛的刀具模型（已排除剑类）...")
final_uids = []

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
    
    # 2. 判断逻辑：必须包含刀，且绝对不能包含剑
    has_knife = any(k in full_text for k in must_have_keywords)
    has_sword = any(k in full_text for k in exclude_keywords)
    
    if has_knife and not has_sword:
        final_uids.append(uid)

# ==========================================
# 结果输出与保存
# ==========================================
print("="*60)
print(f"筛选完成！")
print(f"  - 找到的纯刀具总数 (无剑版): {len(final_uids)}")
print("="*60)

# 保存UID列表
output_file = "all_knife_no_sword_uids.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_uids, f)
print(f"\nUID列表已保存至: {output_file}")

# 打印前10个模型预览
if final_uids:
    print("\n前10个模型预览:")
    for i in range(min(10, len(final_uids))):
        uid = final_uids[i]
        data = annotations[uid]
        print(f"  [{i+1}] 名称: {data.get('name')} | UID: {uid[:12]}...")
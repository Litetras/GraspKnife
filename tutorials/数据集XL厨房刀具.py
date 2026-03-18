import objaverse
import json

print("正在加载 Objaverse XL 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 核心筛选逻辑配置
# ==========================================

# 【白名单】模型描述中必须包含以下词之一 (我们只要这些)
include_keywords = {
    'kitchen knife', 'kitchen', 'chef knife', 'chef', 
    'cooking', 'cleaver', 'fruit knife', 'paring knife',
    'bread knife', 'cutting board', 'utensil', 'cutlery'
}

# 【黑名单】模型描述中绝对不能包含以下词 (有这些的直接扔掉)
exclude_keywords = {
    'sword', 'dagger', 'weapon', 'military', 'tactical',
    'katana', 'samurai', 'survival', 'hunting', 'bayonet',
    'switchblade', 'butterfly', 'karambit', 'ninja',
    'fantasy', 'medieval', 'army', 'gun', 'rifle'
}

print("正在筛选【厨房刀具】并排除【武器】...")

final_uids = []

for uid, item in annotations.items():
    # 1. 获取并拼接文本
    name = item.get('name', '').lower()
    description = item.get('description', '').lower()
    
    # 解析 Tags (兼容字典格式)
    tag_strings = []
    for tag in item.get('tags', []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get('name', '')).lower())
        else:
            tag_strings.append(str(tag).lower())
    tags_text = ' '.join(tag_strings)
    
    full_text = f"{name} {description} {tags_text}"
    
    # 2. 逻辑判断
    # 条件A: 必须包含白名单中的至少一个词
    has_include = any(k in full_text for k in include_keywords)
    # 条件B: 绝对不能包含黑名单中的任何词
    has_exclude = any(k in full_text for k in exclude_keywords)
    
    if has_include and not has_exclude:
        final_uids.append(uid)

# ==========================================
# 结果输出与保存
# ==========================================

print("="*50)
print(f"筛选完成！")
print(f"  - 原始刀具总数 (之前的结果): ~2515")
print(f"  - 过滤后厨房刀具数量: {len(final_uids)}")
print("="*50)

# 保存 UID 列表到本地文件，方便下次直接用
output_file = "kitchen_knife_uids.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_uids, f)

print(f"\nUID 列表已保存至: {output_file}")

# 打印前 3 个看看效果
if final_uids:
    print("\n前 3 个候选模型预览:")
    for i in range(min(3, len(final_uids))):
        uid = final_uids[i]
        data = annotations[uid]
        print(f"  [{i+1}] Name: {data.get('name')} (UID: {uid})")

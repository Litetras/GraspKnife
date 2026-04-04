import objaverse
import json

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 核心过滤逻辑：专注床刷/除尘刷 (Bed Brush / Dust Brush)
# ==========================================

# 【核心基础词】
core_keyword = 'brush'

# 【强相关场景词】必须包含以下至少一个词，确保是清洁、除尘用的手持刷
context_keywords = {
    'dust', 'bed', 'hand', 'sweep', 'clean', 'counter', 'bench', 'dustpan'
}

# 【拒绝的黑名单词】全网封杀牙刷、画笔、化妆刷、梳子、马桶刷等异形刷
exclude_keywords = {
    'tooth', 'paint', 'hair', 'makeup', 'cosmetic', 'toilet', 
    'bottle', 'wire', 'shoe', 'shaving', 'art', 'draw', 'comb',
    'sculpt', 'mascara', 'toothbrush', 'paintbrush', 'hairbrush'
}

print("正在搜集除尘刷/床刷模型（已严格排除牙刷、画笔等无关类别）...")
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
    
    # 2. 判断逻辑
    has_brush = core_keyword in full_text
    has_context = any(k in full_text for k in context_keywords)
    has_excluded = any(k in full_text for k in exclude_keywords)
    
    # 必须是刷子 + 符合除尘/清洁场景 + 绝对不能是画笔/牙刷等
    if has_brush and has_context and not has_excluded:
        final_uids.append(uid)

# ==========================================
# 结果输出与保存
# ==========================================
print("="*60)
print(f"筛选完成！")
print(f"  - 找到的单侧除尘/床刷总数: {len(final_uids)}")
print("="*60)

# 保存UID列表
output_file = "dust_bed_brush_uids.json"
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
import objaverse
import json

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 核心过滤逻辑：灵活碰撞 + 铁血排异
# ==========================================

# 1. 核心词（必须有）
core_keyword = 'brush'

# 2. 场景白名单（必须有至少一个）：聚焦于桌面、手持、扫灰
context_keywords = {
    'dust', 'sweep', 'clean', 'bed', 'desk', 'table', 'counter', 'hand', 'clothes', 'lint'
}

# 3. 终极黑名单（绝对不能有！一旦出现直接枪毙）
exclude_keywords = {
    # 个人护理与艺术 (防牙刷、化妆刷、画笔)
    'paint', 'hair', 'makeup', 'cosmetic', 'toilet', 'bottle', 'wire', 
    'shoe', 'shaving', 'art', 'draw', 'comb', 'sculpt', 'mascara', 'tooth',
    # 厨房卫浴与水洗 (防洗碗刷、浴缸刷、海绵)
    'scrub', 'dish', 'kitchen', 'wash', 'sponge', 'sink', 'bath', 'shower', 'tub', 'laundry',
    # 家电与工业附件 (防吸尘器吸头、扫地机器人、电钻刷头、滚刷)
    'vacuum', 'robot', 'roomba', 'roller', 'spin', 'rotary', 'industrial', 
    'machine', 'motor', 'electric', 'attachment', 'drill', 'cnc', 'pipe',
    # 长柄与大型清洁 (防扫街车、长柄大扫把、洗车大刷、拖把)
    'floor', 'street', 'car', 'mop', 'broomstick', 'yard', 'outdoor'
}

print("正在搜集除尘刷（灵活碰撞词汇，并施加最严苛的黑名单）...")
final_uids = []

for uid, item in annotations.items():
    # 1. 文本提取与格式化
    name = str(item.get('name', '')).lower()
    description = str(item.get('description', '')).lower()
    
    tag_strings = []
    for tag in item.get('tags', []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get('name', '')).lower())
        else:
            tag_strings.append(str(tag).lower())
    tags_text = ' '.join(tag_strings)
    
    # 组合全部文本进行判定，确保是全小写
    full_text = f"{name} {description} {tags_text}"
    
    # 2. 判断逻辑
    # 条件A：必须包含基础词 brush
    has_brush = core_keyword in full_text
    
    # 条件B：必须包含至少一个清扫/桌面的场景词汇
    has_context = any(k in full_text for k in context_keywords)
    
    # 条件C：绝对不能包含黑名单里的任何词汇
    has_excluded = any(k in full_text for k in exclude_keywords)
    
    # 额外保护：如果只有 sweep 这个词，为了防止是扫把(broom)，强行要求包含 brush
    if has_brush and has_context and not has_excluded:
        final_uids.append(uid)

# ==========================================
# 结果输出与保存
# ==========================================
print("\n" + "="*60)
print(f"筛选完成！")
print(f"  - 找到的高纯度除尘/床刷总数: {len(final_uids)}")
print("="*60)

output_file = "pure_dust_brush_uids.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_uids, f)
print(f"\nUID列表已保存至: {output_file}")

# 打印前 10 个模型预览
if final_uids:
    print("\n前 10 个模型预览:")
    for i in range(min(10, len(final_uids))):
        uid = final_uids[i]
        data = annotations[uid]
        print(f"  [{i+1}] 名称: {data.get('name')} | UID: {uid[:12]}...")
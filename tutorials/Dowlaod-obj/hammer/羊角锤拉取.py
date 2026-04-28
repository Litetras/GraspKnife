import objaverse
import json

print("正在加载 Objaverse 标注库...")
annotations = objaverse.load_annotations()

# ==========================================
# 核心过滤逻辑：主词 + 上下文交叉验证
# ==========================================

# 1. 明确的专属词（只要有这些，大概率是目标）
exact_keywords = {'claw hammer', 'clawhammer', 'claw-hammer'}

# 2. 宽泛的主词
general_keywords = {'hammer'}

# 3. 必须具备的【上下文环境词】（用来证明它是一个工具，而不是武器或玩具）
# 如果它只叫 hammer，那就必须包含以下至少一个词
context_keywords = {
    'tool', 'tools', 'hardware', 'carpenter', 'woodworking', 
    'construction', 'workshop', 'repair', 'equipment', 'nail'
}

# 4. 超级黑名单（排除奇幻、武器、生物、重锤等）
exclude_keywords = {
    # 武器与奇幻
    'war', 'weapon', 'thor', 'mjolnir', 'banhammer', 'fantasy', 'magic', 'smash',
    'sword', 'axe', 'knight', 'inquisitor', 'forge', 'blacksmith',
    # 其他类型的锤/工具
    'sledge', 'jackhammer', 'mallet', 'gavel', 'judge', 'drill',
    # 生物干扰（防止 claw 匹配到爪子，nail 匹配到指甲）
    'shark', 'animal', 'monster', 'creature', 'dragon', 'dinosaur', 
    'bird', 'cat', 'paw', 'finger', 'hand', 'character', 'robot'
}

print("正在通过【上下文交叉验证】搜集工具锤模型...")
final_uids = []

for uid, item in annotations.items():
    name = str(item.get('name', '')).lower()
    description = str(item.get('description', '')).lower()
    
    tag_strings = []
    for tag in item.get('tags', []):
        if isinstance(tag, dict):
            tag_strings.append(str(tag.get('name', '')).lower())
        else:
            tag_strings.append(str(tag).lower())
    tags_text = ' '.join(tag_strings)
    
    full_text = f"{name} {description} {tags_text}"
    
    # --- 开始判断 ---
    
    # 1. 碰触黑名单，直接死刑
    if any(k in full_text for k in exclude_keywords):
        continue
        
    # 2. 检查主词和上下文
    has_exact = any(k in full_text for k in exact_keywords)
    has_general = any(k in full_text for k in general_keywords)
    has_context = any(k in full_text for k in context_keywords)
    
    # 录用条件：
    # 条件A：明确写了是羊角锤 (claw hammer)
    # 条件B：写了是锤子 (hammer)，并且带有工具属性 (tool/hardware 等)
    if has_exact or (has_general and has_context):
        final_uids.append(uid)

# ==========================================
# 结果输出与保存
# ==========================================
print("="*60)
print(f"筛选完成！")
print(f"  - 找到的高质量工具锤总数: {len(final_uids)}")
print("="*60)

# 保存UID列表
output_file = "filtered_tool_hammers_uids.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_uids, f)
print(f"\nUID列表已保存至: {output_file}")
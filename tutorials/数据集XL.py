import objaverse

print("正在加载 Objaverse XL 标注库...")
annotations = objaverse.load_annotations()

target_keywords = {'knife'}

print("正在统计刀具模型数量...")
matching_uids = []

for uid, item in annotations.items():
    # 1. 安全获取名称和描述
    name = item.get('name', '') or ''
    description = item.get('description', '') or ''
    
    # 2. 修复 Tags 解析逻辑
    # Objaverse XL 的 tags 可能是字典列表，也可能是字符串列表，这里做兼容处理
    tag_strings = []
    tags_raw = item.get('tags', [])
    if tags_raw:
        for tag in tags_raw:
            if isinstance(tag, dict):
                # 如果是字典，尝试提取 'name' 或 'value' 字段
                tag_strings.append(str(tag.get('name', tag.get('value', ''))))
            elif isinstance(tag, str):
                tag_strings.append(tag)
    
    tags_text = ' '.join(tag_strings)
    
    # 3. 合并文本并匹配
    full_text = f"{name} {description} {tags_text}".lower()
    
    if any(keyword in full_text for keyword in target_keywords):
        matching_uids.append(uid)

print("="*40)
print(f"Objaverse XL 中匹配到的模型总数: {len(matching_uids)}")
if matching_uids:
    print(f"前 5 个模型的 UID 示例: {matching_uids[:5]}")
    
    # 可选：打印其中一个模型的完整元数据看看长啥样
    sample_uid = matching_uids[0]
    print(f"\n示例模型元数据 ({sample_uid}):")
    print(annotations[sample_uid])
print("="*40)

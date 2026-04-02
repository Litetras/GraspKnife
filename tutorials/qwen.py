import ollama

res = ollama.chat(
    model='qwen3.5:4b',
    messages=[{
        'role': 'user',
        'content': '告诉我这张图片里有什么？',
        'images': ['/home/zyp/Pictures/Screenshots/Screenshot from 2026-03-26 22-14-14.png']
    }]
)
print(res['message']['content'])#
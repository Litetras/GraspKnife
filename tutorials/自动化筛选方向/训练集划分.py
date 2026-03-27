# 定义固定的后缀顺序
suffixes = ['_up.obj', '_down.obj', '_top.obj', '_low.obj']

# 循环生成 1 到 135 的所有文件名
for num in range(1, 136):  # range左闭右开，136才能包含135
    for suffix in suffixes:
        print(f"kitchen_knife_{num}{suffix}")

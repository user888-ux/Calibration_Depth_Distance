import matplotlib.pyplot as plt

# 从 result.txt 读取数据
file_path = "result.txt"
right_values = []

try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue          # 跳过空行或不符合格式的行
            right_part = line.split('->')[1].strip()
            if right_part.endswith('m'):
                right_part = right_part[:-1]   # 去掉末尾的单位 'm'
            try:
                value = float(right_part)
                right_values.append(value)
            except ValueError:
                print(f"警告：无法转换行 -> {line}")
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请确保 result.txt 在当前目录下。")
    exit()

# 打印提取的数据统计
print("提取的右侧数值（共 {} 个）：".format(len(right_values)))
if right_values:
    print("前10个值:", right_values[:10])
else:
    print("没有有效数据，无法绘图。")
    exit()

# 绘制散点图
x = list(range(len(right_values)))
plt.figure(figsize=(10, 5))
plt.scatter(x, right_values, color='blue', alpha=0.7, s=50)
plt.xlabel('行索引')
plt.ylabel('右侧数值 (m)')
plt.title('右侧数值散点图')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
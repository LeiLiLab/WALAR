import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置全局样式 - 更专业的学术风格
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 优先使用Arial，学术论文常用
plt.rcParams['axes.unicode_minus'] = False

# 使用seaborn样式，更现代美观
sns.set_style("whitegrid")
sns.set_palette("husl")

# 调整模型顺序：untrained最左边，mixed最右边
models = ['Untrained', 'En-X', 'Ar-X', 'Tr-X', 'Hi-X', 'Mixed']
language_directions = ['En-X', 'Ar-X', 'Tr-X', 'Hi-X', 'Avg']

# 调整数据顺序以匹配新的模型顺序
data = np.array([
    [10.28, 6.65, 6.88, 7.8, 7.9],     # 未训练模型 - 现在在最左边
    [12.42, 8.09, 8.35, 9.24, 9.53],   # En-X模型
    [12.35, 8.46, 8.62, 9.27, 9.68],   # Ar-X模型
    [12.48, 8.37, 8.9, 9.49, 9.81],    # Tr-X模型
    [12.3, 7.97, 8.39, 9.38, 9.51],    # Hi-X模型
    [12.7, 8.67, 9.08, 9.71, 10.04],   # 混合模型 - 现在在最右边
])

# 创建图形和坐标轴 - 调整尺寸比例
fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)  # 进一步增加高度为8.5，给图例更多空间

# 设置更明亮的颜色方案 - 使用seaborn的明亮调色板
colors = ['#937860', '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

# 调整柱状图宽度和位置
bar_width = 0.14
x_pos = np.arange(len(language_directions))

# 绘制柱状图 - 添加透明度使图表更柔和
bars = []
for i, model in enumerate(models):
    bar = ax.bar(x_pos + i * bar_width, data[i], bar_width, 
                 label=model, color=colors[i], edgecolor='white', 
                 linewidth=1.2, alpha=0.9, zorder=3)
    bars.append(bar)

# 添加数值标签 - 优化位置和样式，避免太紧凑
for i, model_data in enumerate(data):
    for j, value in enumerate(model_data):
        # 根据数值大小调整标签位置，增加垂直偏移量避免紧凑
        vertical_offset = 0.25 if value < 10 else 0.35
        # 对于特别小的数值，进一步调整位置
        if value < 7:
            vertical_offset = 0.2
        ax.text(j + i * bar_width, value + vertical_offset, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2C3E50')

# 设置坐标轴和标签 - 优化字体和样式
ax.set_xlabel('Language Direction', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('spBLEU', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Generalization of RL', 
             fontsize=14, fontweight='bold', pad=20)

# 设置x轴刻度
ax.set_xticks(x_pos + bar_width * 2.5)
ax.set_xticklabels(language_directions, fontsize=11, fontweight='bold')

# 设置y轴刻度 - 修改为从4开始
ax.set_yticks(np.arange(4, 15, 2))  # 从4开始，步长为2，到14结束
ax.tick_params(axis='y', labelsize=10)

# 优化图例 - 进一步向上移动
ax.legend(loc='upper center', fontsize=11, 
          frameon=True, fancybox=True, shadow=True, framealpha=0.9,
          bbox_to_anchor=(0.5, 1.05),  # 将y坐标从0.98调整为1.05，进一步向上移动
          ncol=3)  # 分成3列显示，更紧凑

# 优化网格线
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
ax.grid(True, axis='x', alpha=0.1, linestyle='-', linewidth=0.5)

# 设置y轴范围，从4开始而不是0 - 这是主要的修改
ax.set_ylim(4, 14.5)  # 将下限从0改为4

# 设置背景色为更柔和的白色
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# 添加边框美化
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#BDC3C7')

# 移除了Avg列的垂直分隔线和标注

# 调整布局，确保所有元素都显示
plt.tight_layout(rect=[0, 0, 1, 0.92])  # 将顶部空间从0.95调整为0.92，为图例留出更多空间

# 显示图形
# plt.show()

# 可选：保存为高分辨率图片（论文需要）
plt.savefig('/mnt/gemini/data1/yifengliu/qe-lr/figure/model_performance_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('model_performance_comparison.pdf', bbox_inches='tight')  # 矢量图格式，论文推荐
# plt.savefig("/mnt/gemini/data1/yifengliu/qe-lr/figure/model_performance_comparison.png")
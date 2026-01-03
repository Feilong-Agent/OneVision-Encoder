import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ======================
# 1. 数据（示例，可替换）
# ======================
benchmarks = [
    "SSV2", "Diving48", "Perception Test", "CharadesEgo",
    "Epic_Verb", "Epic_Noun", "K400", "HMDB51"
]

# 每个 benchmark 对应 5 个模型的分数
scores = {
    "MetaCLIP2": [45.9, 30.7, 46.4, 10.2, 44.4, 36.6, 79.1, 77.0],
    "DINOv3": [56.1, 50.7, 58.1, 12.4, 60.0, 47.9, 81.4, 81.4],
    "SigLIP2": [57.9, 53.2, 56.3, 12.8, 58.8, 45.2, 81.3, 82.1],
    "AIMv2": [53.8, 45.3, 52.9, 11.5, 54.4, 42.7, 79.0, 81.5],
    "OV-Encoder": [57.0, 56.0, 57.7, 12.4, 61.9, 53.3, 84.1, 82.2],
}

models = list(scores.keys())
num_models = len(models)
num_bench = len(benchmarks)

# ======================
# 2. 统一配色（重点）
# ======================
colors = {
    "OV-Encoder": "#5B5BD6",   # 蓝紫 (OV-Encoder是蓝色)
    "MetaCLIP2": "#B7DDB0",    # 浅绿
    "DINOv3": "#F6D8B8",       # 米色
    "SigLIP2": "#F39AC1",      # 粉色
    "AIMv2": "#C8A2C8",        # 淡紫色
}

# ======================
# 3. 极坐标布局
# ======================
angles = np.linspace(0, 2 * np.pi, num_bench, endpoint=False)
bar_width = 2 * np.pi / num_bench / (num_models + 1)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 去掉边框和网格线
ax.spines['polar'].set_visible(False)  # 去掉圆形边框
ax.grid(False)  # 去掉网格线

# 设置径向范围，从较大的值开始，缩短柱状图
radial_start = 35  # 柱状图起始位置（内圈半径）
radial_end = 75    # 柱状图结束位置（缩短长度）

# 计算所有数据的最小值和最大值，用于智能缩放
all_values = [v for values_list in scores.values() for v in values_list]
min_val = min(all_values)
max_val = max(all_values)
value_range = max_val - min_val

# 使用基线缩放：将最小值映射到30%的高度，最大值映射到100%
# 这样即使小数值也能清晰显示
baseline_ratio = 0.3  # 最小值显示为30%高度

# ======================
# 4. 画 bars（缩短后的柱状图）
# ======================
for i, model in enumerate(models):
    values = scores[model]
    offset = (i - (num_models - 1) / 2) * bar_width
    
    # 将原始分数映射到缩短的范围，使用基线缩放
    # 公式：scaled = baseline_ratio + (value - min) / (max - min) * (1 - baseline_ratio)
    if value_range > 0:
        normalized_values = (np.array(values) - min_val) / value_range
    else:
        # 如果所有值相同，使用中间值
        normalized_values = np.ones_like(values) * 0.5
    scaled_values = (baseline_ratio + normalized_values * (1 - baseline_ratio)) * (radial_end - radial_start)

    bars = ax.bar(
        angles + offset,
        scaled_values,
        width=bar_width * 0.9,
        bottom=radial_start,  # 从radial_start开始，而不是从0
        color=colors[model],
        edgecolor="white",  # 浅色边框
        linewidth=0.5,
        alpha=0.95,
        label=model
    )

    # 数值标注（在柱状图内部顶端，沿着柱状方向）
    for j, (angle_with_offset, val, scaled_val) in enumerate(zip(angles + offset, values, scaled_values)):
        # 将数值放在柱状图顶端（接近radial_start + scaled_val）
        text_radius = radial_start + scaled_val * 0.85  # 在顶端85%位置
        
        # 根据模型选择文字颜色：OV-Encoder用白色，其他用黑色
        if model == "OV-Encoder":
            text_color = 'white'
        else:
            text_color = 'black'
        
        # 使用基础角度（不加offset）来确保同一数据集的所有数值朝向一致
        base_angle = angles[j]
        text_rotation = np.degrees(base_angle)
        
        # 调整旋转方向使文字始终可读，所有同一数据集的数值保持相同朝向
        if 90 < text_rotation < 270:
            text_rotation = text_rotation - 90  # 向内旋转
        else:
            text_rotation = text_rotation + 90  # 向外旋转
        
        ax.text(
            angle_with_offset,  # 位置使用带offset的角度
            text_radius,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=6.5,
            rotation=text_rotation,  # 旋转使用基础角度
            rotation_mode="anchor",
            color=text_color,
            fontweight='bold'
        )

# ======================
# 5. benchmark 标签（放在内圈）
# ======================
ax.set_xticks(angles)
# 将标签设置为空，我们手动添加到内圈
ax.set_xticklabels([])

# 先不添加标签，等白色圆圈画完后再添加（确保标签在上层）

ax.set_yticks([])
ax.set_ylim(0, radial_end + 10)  # 调整y轴范围

# ======================
# 6. 中心留白（放 logo）- 增大内圈
# ======================
# 增大中心圆，覆盖更大区域
circle = plt.Circle((0, 0), radial_start - 3, transform=ax.transData._b, color="white", zorder=10)
ax.add_artist(circle)

# 加载并显示图片在中间
# 注意：需要在当前目录下提供 intro_tem.jpg 图片文件

try:
    img = plt.imread("intro_tem.jpg")
    # 增大zoom以让中心图片更大
    imagebox = OffsetImage(img, zoom=0.25)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False, zorder=11)
    ax.add_artist(ab)
except (FileNotFoundError, OSError):
    # 如果图片加载失败，显示备用文本
    ax.text(
        0, 0,
        "OneVision\nEncoder",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        zorder=11
    )

# 现在在白色圆圈外围添加标签（在柱状图起始位置）
# 为每个数据集添加弧线标记和弧形文字
arc_radius = radial_start - 6  # 弧线位置
label_radius = radial_start - 9  # 文字在弧线下方

for angle, benchmark in zip(angles, benchmarks):
    # 计算每个benchmark占据的角度范围
    arc_width = 2 * np.pi / num_bench * 0.7  # 弧线宽度
    
    # 绘制弧线
    arc_angles = np.linspace(angle - arc_width/2, angle + arc_width/2, 50)
    arc_x = arc_angles
    arc_y = np.full_like(arc_angles, arc_radius)
    ax.plot(arc_x, arc_y, color='#999999', linewidth=2, alpha=0.5, zorder=12)
    
    # 将文字按弧形排列（每个字符单独放置）
    text_len = len(benchmark)
    if text_len > 0:
        # 计算字符间距，使文字沿着弧线分布
        char_arc_width = arc_width * 0.75  # 字符占用的弧线宽度
        char_angles = np.linspace(angle - char_arc_width/2, angle + char_arc_width/2, text_len)
        
        for char_angle, char in zip(char_angles, benchmark):
            # 计算每个字符的旋转角度
            rotation = np.degrees(char_angle)
            
            # 调整文字旋转使其沿着弧线方向，且可读
            if 90 < rotation < 270:
                rotation = rotation + 180
                va = 'top'
            else:
                va = 'bottom'
            
            ax.text(
                char_angle,
                label_radius,
                char,
                ha='center',
                va=va,
                fontsize=7.5,
                rotation=rotation,
                rotation_mode="anchor",
                fontweight='bold',
                color='#444444',
                zorder=13,
                family='sans-serif'
            )


# ======================
# 7. 图例
# ======================
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig("fig1.png", dpi=300, bbox_inches="tight")
plt.show()

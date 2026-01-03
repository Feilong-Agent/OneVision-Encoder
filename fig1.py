import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ======================
# 1. 数据（示例，可替换）
# ======================
benchmarks = [
    "MathVista", "MathVision", "MMMU", "MME",
    "VideoMME", "LCBenchV6", "AI2D",
    "MMMU-Pro", "RealWorldQA", "ERA",
    "DesignCode", "MMBench"
]

# 每个 benchmark 对应 4 个模型的分数
scores = {
    "Model-A": [50.9, 66.5, 57.0, 79.2, 79.2, 54.3, 74.7, 84.9, 92.0, 51.3, 79.2, 57.0],
    "Model-B": [74.5, 57.7, 51.2, 80.9, 80.6, 51.8, 70.3, 77.7, 90.3, 50.3, 76.0, 51.2],
    "Model-C": [77.7, 66.0, 48.1, 78.7, 77.3, 45.2, 46.6, 74.5, 88.9, 42.0, 77.3, 48.1],
    "Model-D": [84.9, 45.8, 42.4, 74.4, 73.3, 44.6, 33.9, 50.9, 85.3, 28.0, 68.5, 42.4],
}

models = list(scores.keys())
num_models = len(models)
num_bench = len(benchmarks)

# ======================
# 2. 统一配色（重点）
# ======================
colors = {
    "Model-A": "#5B5BD6",   # 蓝紫
    "Model-B": "#B7DDB0",   # 浅绿
    "Model-C": "#F6D8B8",   # 米色
    "Model-D": "#F39AC1",   # 粉色
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

# ======================
# 4. 画 bars（缩短后的柱状图）
# ======================
for i, model in enumerate(models):
    values = scores[model]
    offset = (i - (num_models - 1) / 2) * bar_width
    
    # 将原始分数映射到缩短的范围
    scaled_values = (np.array(values) / 100.0) * (radial_end - radial_start)

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

    # 数值标注（在柱状图内部，沿着柱状方向）
    for angle, val, scaled_val in zip(angles + offset, values, scaled_values):
        # 将数值放在柱状图中间位置
        text_radius = radial_start + scaled_val / 2
        ax.text(
            angle,
            text_radius,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=6.5,
            rotation=np.degrees(angle),
            rotation_mode="anchor",
            color='white',
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
    # 调整zoom以适配更大的中心区域，稍微缩小避免遮挡标签
    imagebox = OffsetImage(img, zoom=0.18)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False, zorder=11)
    ax.add_artist(ab)
except (FileNotFoundError, OSError):
    # 如果图片加载失败，显示备用文本
    ax.text(
        0, 0,
        "OneVision\nEncoder",
        ha="center",
        va="center",
        fontsize=16,
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
    
    # 沿着弧线放置文字（弧形排列）
    text_len = len(benchmark)
    if text_len > 0:
        # 为每个字符计算位置，使字符间距更紧密
        char_spacing = arc_width * 0.6 / max(text_len, 1)
        start_angle = angle - (text_len - 1) * char_spacing / 2
        
        for i, char in enumerate(benchmark):
            char_angle = start_angle + i * char_spacing
            rotation = np.degrees(char_angle)
            
            # 调整文字旋转使其可读
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
                zorder=13
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

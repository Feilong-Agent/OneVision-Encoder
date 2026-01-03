import numpy as np
import matplotlib.pyplot as plt

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

# ======================
# 4. 画 bars
# ======================
for i, model in enumerate(models):
    values = scores[model]
    offset = (i - (num_models - 1) / 2) * bar_width

    bars = ax.bar(
        angles + offset,
        values,
        width=bar_width * 0.9,
        color=colors[model],
        edgecolor="none",
        alpha=0.95,
        label=model
    )

    # 数值标注
    for angle, val in zip(angles + offset, values):
        ax.text(
            angle,
            val + 2,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            rotation=np.degrees(angle),
            rotation_mode="anchor"
        )

# ======================
# 5. benchmark 标签
# ======================
ax.set_xticks(angles)
ax.set_xticklabels(benchmarks, fontsize=10)

ax.set_yticks([])
ax.set_ylim(0, 100)

# ======================
# 6. 中心留白（放 logo）
# ======================
circle = plt.Circle((0, 0), 15, transform=ax.transData._b, color="white", zorder=10)
ax.add_artist(circle)

ax.text(
    0, 0,
    "Qwen",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    zorder=11
)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

img = plt.imread("intro_tem.jpg")
imagebox = OffsetImage(img, zoom=0.15)
ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
ax.add_artist(ab)


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

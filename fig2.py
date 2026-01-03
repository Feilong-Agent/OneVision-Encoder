import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ======================
# 1. 数据（Video和Image任务）
# ======================
# Video任务的benchmarks
video_benchmarks = [
    "MVBench", "MLVU-dev", "NExT-QA (MC)", "VideoMME",
    "Perception Test", "TOMATO", "LongVideoBenc-Val-Video"
]

# Image任务的benchmarks
image_benchmarks = [
    "ChartQA", "DocVQA", "InfoVQA", "MMBench-EN",
    "OCRBench", "OCRBench v2", "MMStar", "RealWorldQA"
]

# 合并所有benchmarks
benchmarks = video_benchmarks + image_benchmarks

# 每个 benchmark 对应的模型分数（使用Qwen3-4B的数据）
scores = {
    "OV-ViT-Codec": [51.9, 44.7, 74.6, 54.6, 60.4, 21.2, 50.4, 78.6, 79.7, 44.8, 78.5, 617, None, 52.8, 61.7],
    "OV-ViT": [49.8, 49.4, 71.9, 49.3, 56.7, 21.8, 45.5, 77.8, 79.5, 45.5, 78.5, 630.0, 26.1, 54.3, 61.2],
    "SigLIP2": [47.2, 48.4, 70.6, 46.8, 56.0, 22.3, 45.2, 76.4, 75.0, 42.0, 79.6, 621.0, 26.1, 55.0, 62.1],
}

# Note: OCRBench和OCRBench v2使用原始数值，其他都是百分比
# 我们需要归一化OCRBench的值以便可视化

models = list(scores.keys())
num_models = len(models)
num_bench = len(benchmarks)

# ======================
# 2. 统一配色（与fig1保持一致的风格）
# ======================
colors = {
    "OV-ViT-Codec": "#5B5BD6",   # 蓝紫 (主要方法)
    "OV-ViT": "#B7DDB0",          # 浅绿
    "SigLIP2": "#F39AC1",         # 粉色
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
# 归一化OCRBench数据：将600-630范围映射到0-100
all_values = []
for model_name, values_list in scores.items():
    for i, v in enumerate(values_list):
        if v is not None:
            # OCRBench (index 11) 归一化
            if i == 11:
                normalized_v = (v - 500) / 2  # 500->0, 700->100
                all_values.append(normalized_v)
            else:
                all_values.append(v)

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
    values_raw = scores[model]
    
    # 归一化处理
    values = []
    for j, v in enumerate(values_raw):
        if v is None:
            values.append(0)  # None值用0代替，后面不显示
        elif j == 11:  # OCRBench归一化
            values.append((v - 500) / 2)
        else:
            values.append(v)
    
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
    for j, (angle_with_offset, val_raw, scaled_val) in enumerate(zip(angles + offset, values_raw, scaled_values)):
        if val_raw is None:
            continue  # 跳过None值
            
        # 将数值放在柱状图顶端（接近radial_start + scaled_val）
        text_radius = radial_start + scaled_val * 0.85  # 在顶端85%位置
        
        # 根据模型选择文字颜色：OV-ViT-Codec用白色，其他用黑色
        if model == "OV-ViT-Codec":
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
        
        # 显示原始值
        if j == 11:  # OCRBench显示原始分数
            display_val = f"{int(val_raw)}"
        elif j == 12:  # OCRBench v2
            display_val = f"{val_raw:.1f}"
        else:
            display_val = f"{val_raw:.1f}"
        
        ax.text(
            angle_with_offset,  # 位置使用带offset的角度
            text_radius,
            display_val,
            ha="center",
            va="center",
            fontsize=6.5,
            rotation=text_rotation,  # 旋转使用基础角度
            rotation_mode="anchor",
            color=text_color,
            fontweight='bold'
        )

# ======================
# 5. 配置极坐标
# ======================
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0, radial_end + 10)

# ======================
# 6. 中间白色圆圈 + 图片
# ======================
white_circle = plt.Circle(
    (0, 0),
    radial_start - 3,
    transform=ax.transData._b,
    color="white",
    zorder=10
)
ax.add_artist(white_circle)

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

# 数据集标签已注释掉，用户将在PPT中手动添加
# (与fig1保持一致，方便PPT集成)

# ======================
# 7. 图例（排成一行）
# ======================
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,  # 3个模型排成一行
    frameon=False
)

plt.tight_layout()
plt.savefig("fig2.png", dpi=300, bbox_inches="tight")
plt.show()

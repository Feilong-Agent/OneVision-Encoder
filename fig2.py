import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ======================
# 1. 数据（Video、Image和Document任务）
# ======================
# 重新组织布局：Video (left) -> Document (top/center) -> Image (right)
# 这样Document在圆弧顶部中间，Video和Image在两侧

# Video任务的benchmarks (7个)
video_benchmarks = [
    "MVBench", "MLVU-dev", "NExT-QA (MC)", "VideoMME",
    "Perception Test", "TOMATO", "LongVideoBenc-Val-Video"
]

# Document任务的benchmarks (2个) - 放在中间
document_benchmarks = [
    "OCRBench", "OCRBench v2"
]

# Image任务的benchmarks (6个) - 合并所有Image任务
image_benchmarks = [
    "ChartQA", "DocVQA", "InfoVQA", "MMBench-EN", 
    "MMStar", "RealWorldQA"
]

# 合并所有benchmarks: Video + Document + Image
benchmarks = video_benchmarks + document_benchmarks + image_benchmarks

# 每个 benchmark 对应的模型分数（使用Qwen3-4B的数据）
# 数据顺序：Video (7) + Document (2) + Image (6)
scores = {
    "OV-Encoder": [
        # Video (7个)
        51.9, 44.7, 74.6, 54.6, 60.4, 21.2, 50.4,
        # Document (2个)
        617, None,
        # Image (6个)
        78.6, 79.7, 44.8, 78.5, 52.8, 61.7
    ],
    "SigLIP2": [
        # Video (7个)
        47.2, 48.4, 70.6, 46.8, 56.0, 22.3, 45.2,
        # Document (2个)
        621.0, 26.1,
        # Image (6个)
        76.4, 75.0, 42.0, 79.6, 55.0, 62.1
    ],
}

# Note: OCRBench使用原始数值（600+范围），其他都是百分比
# 我们需要归一化OCRBench的值以便可视化

models = list(scores.keys())
num_models = len(models)
num_bench = len(benchmarks)

# ======================
# 2. 统一配色（与fig1保持一致的风格）
# ======================
colors = {
    "OV-Encoder": "#5B5BD6",   # 蓝紫 (主要方法，与fig1的OV-Encoder一致)
    "SigLIP2": "#F39AC1",      # 粉色
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
# 新布局中OCRBench在index 7位置
all_values = []
for model_name, values_list in scores.items():
    for i, v in enumerate(values_list):
        if v is not None:
            # OCRBench (index 7 in new layout) 归一化
            if i == 7:
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
        elif j == 7:  # OCRBench归一化 (新布局中的位置)
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
        
        # 显示原始值
        if j == 7:  # OCRBench显示原始分数 (新布局中的位置)
            display_val = f"{int(val_raw)}"
        elif j == 8:  # OCRBench v2 (新布局中的位置)
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
# 5.5. 数据集标签（弧形摆放）和类别图标
# ======================
# 添加弧形数据集标签
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

# 添加类别分隔（使用半弧线）
# 新布局：Video (0-7) -> Document (7-9) -> Image (9-15)
# Document在圆弧顶部/中间，Video和Image在两侧

# 定义类别边界和颜色（使用更淡的颜色避免冲突）
categories = [
    {"name": "Video", "start": 0, "end": 7, "color": "#A8B3E8"},      # 淡蓝色 - 左侧
    {"name": "Document", "start": 7, "end": 9, "color": "#D9D9D9"},   # 淡灰色 - 中间/顶部
    {"name": "Image", "start": 9, "end": 15, "color": "#F9D5E5"},     # 淡粉色 - 右侧（合并所有Image）
]

# 绘制类别分隔半弧（放在圆圈内部，接近数据集标签）- 更窄的范围并填充，增加间隙
arc_outer_radius = 24
arc_inner_radius = 23.5

# 计算单个单元的角度
unit = 2 * np.pi / num_bench

# 方式 A：用比例指定间隙，表示在每个单元之间留多少比例的空白（例如 0.20 -> 20% 的单元宽度）
gap_frac = 0.20
gap = unit * gap_frac

for category in categories:
    start_idx = category["start"]
    end_idx = category["end"]
    color = category["color"]
    name = category["name"]
    
    # 计算收紧后的起止角（每边缩进 gap/2）
    start_angle = angles[start_idx] - unit/2 + gap/2
    end_angle = angles[end_idx - 1] + unit/2 - gap/2
    
    # 绘制半弧线 - 使用填充的弧形区域
    arc_angles = np.linspace(start_angle, end_angle, 100)
    
    # 创建填充的弧形区域（使用fill_between）
    ax.fill_between(arc_angles, 
                     arc_inner_radius, arc_outer_radius,
                     color=color, alpha=0.6, zorder=11)
    
    # 在半弧中心位置添加类别名称（内圈，较小字体）
    center_angle = (start_angle + end_angle) / 2
    text_radius = (arc_inner_radius + arc_outer_radius) / 2  # 在内外弧之间
    
    # 计算文字旋转角度
    text_rotation = np.degrees(center_angle) - 90
    if 90 < np.degrees(center_angle) < 270:
        text_rotation = np.degrees(center_angle) + 90
    
    ax.text(
        center_angle,
        text_radius,
        name,
        ha='center',
        va='center',
        fontsize=7.5,  # 稍微增大字体
        fontweight='bold',
        color=color,
        rotation=text_rotation,
        rotation_mode="anchor",
        zorder=15
    )

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
    ncol=2,  # 2个模型排成一行
    frameon=False
)

plt.tight_layout()
plt.savefig("fig2.png", dpi=300, bbox_inches="tight")
plt.show()

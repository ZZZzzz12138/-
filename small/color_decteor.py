import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体（Windows）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题


def show(img, title="original", cmap=None):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()



def compute_arc_span(cnt, center):
    """计算轮廓中所有点相对于圆心的极角跨度"""
    cx, cy = center
    angles = []
    for pt in cnt:
        x, y = pt[0]
        angle = np.arctan2(y - cy, x - cx)
        angles.append(angle)
    angles = np.unwrap(np.array(angles))  # 解开角度跳变（-π 到 π）
    return np.ptp(angles)  # max - min，表示弧度跨度


def detect_red_or_blue(hue):
    hue = int(hue)
    if hue <= 20 or hue >= 160:
        return "红色"
    elif 70 <= hue <= 140:
        return "蓝色"
    else:
        return "未知"

# 加载图像
img_path = r"D:\Desktop\small\test_images\test7..png"
  # ← 修改为你的图像路径
img = cv2.imread(img_path)
assert img is not None, "图像加载失败！"

show(img, "original")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)
edges = cv2.Canny(enhanced, 100, 200)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

titles = ["灰度图", "中值滤波", "CLAHE增强", "Canny边缘"]
images = [gray, blurred, enhanced, edges]

for i in range(4):
    axs[i].imshow(images[i], cmap="gray")
    axs[i].set_title(titles[i])
    axs[i].axis("off")

plt.show()
height, width = edges.shape
img_center = (width // 2, height // 2)

print(f"图像中心位置：{img_center}")
show(edges, "Step 1：Canny 边缘图", cmap="gray")
# 🔧 函数：最小二乘拟合圆
def fit_circle_least_squares(points):
    pts = np.array(points).reshape(-1, 2).astype(np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return (int(xc), int(yc)), r

# 图像参数
img_h, img_w = img.shape[:2]
img_min_dim = min(img_w, img_h)
min_radius_thresh = img_min_dim * 0.2
max_radius_thresh = img_min_dim * 0.45

# 提取轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 输出图像
output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
qualified_circles = []

# 拟合所有合法圆
for idx, cnt in enumerate(contours):
    if len(cnt) < 5:
        continue
    try:
        center, radius = fit_circle_least_squares(cnt)
        if min_radius_thresh <= radius <= max_radius_thresh:
            qualified_circles.append((idx, cnt, center, radius))
    except:
        continue

# 🔴 找到半径最大的圆
max_idx = -1
max_radius = -1
for i, (_, _, _, radius) in enumerate(qualified_circles):
    if radius > max_radius:
        max_radius = radius
        max_idx = i

# 可视化
for new_idx, (orig_idx, cnt, center, radius) in enumerate(qualified_circles):
    # 原始弧段轮廓（红色）
    cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    # 圆心
    cv2.circle(output, center, 4, (255, 0, 0), -1)  # 蓝点

    # 拟合圆
    if new_idx == max_idx:
        cv2.circle(output, center, int(radius), (0, 0, 255), 2)  # 最大圆红色
    else:
        cv2.circle(output, center, int(radius), (0, 255, 0), 2)  # 其他圆绿色

    # 编号标注（白色）
    mid_pt = tuple(cnt[len(cnt) // 2][0])
    cv2.putText(output, f"{new_idx}", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"编号 #{new_idx}：圆心=({center[0]}, {center[1]}), 半径={radius:.2f}px")

print(f"\n✅ 共拟合并绘制真实圆数量：{len(qualified_circles)}")
print(f"最大圆编号：{max_idx}，半径={max_radius:.2f}px")
print(f"半径筛选范围：{min_radius_thresh:.2f}px ~ {max_radius_thresh:.2f}px")

# 显示图像
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("高亮半径最大圆（红色）")
plt.axis("off")
plt.show()
# 检测圆
best_circles = None
for p2 in [30, 25, 20]:
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(height, width) // 8,
        param1=100,
        param2=p2,
        minRadius=50,
        maxRadius=300
    )
    if circles is not None:
        best_circles = circles
        break

if best_circles is None:
    print("❌ 没有检测到任何圆")
else:
    print(f"✅ 检测到 {best_circles.shape[1]} 个圆")

    # 画出所有候选圆
    temp_img = img.copy()
    circles = np.uint16(np.around(best_circles))
    for circle in circles[0, :]:
        cx, cy, r = circle
        cv2.circle(temp_img, (cx, cy), r, (0, 255, 0), 2)  # 圆边界绿色
        cv2.circle(temp_img, (cx, cy), 3, (0, 0, 255), -1)  # 圆心红色点

    show(temp_img, "Step 2：检测到的所有圆（绿）+ 圆心（红）")
output_img = img.copy()
circles = np.uint16(np.around(best_circles))
best_circle = None
best_score = float('inf')

for circle in circles[0, :]:
    cx, cy, r = circle
    # 可视化所有候选圆
    cv2.circle(output_img, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(output_img, (cx, cy), 3, (0, 0, 255), -1)

    # 计算与图像中心的距离
    dist = np.hypot(cx - img_center[0], cy - img_center[1])
    if dist < best_score:
        best_score = dist
        best_circle = circle

# 高亮最佳圆
if best_circle is not None:
    cv2.circle(output_img, (best_circle[0], best_circle[1]), best_circle[2], (255, 0, 0), 4)

show(output_img, "检测到的所有圆 + 最中心圆（蓝色高亮）")

# 提取圆心和半径
cx, cy, r = best_circle
r_outer = int(r)
r_inner = int(r * 0.85)

# 创建空掩码
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# 绘制外圆为白色
cv2.circle(mask, (cx, cy), r_outer, 255, thickness=-1)
# 抠掉内圆为黑色
cv2.circle(mask, (cx, cy), r_inner, 0, thickness=-1)

# 应用掩码
ring_region = cv2.bitwise_and(img, img, mask=mask)
# 可视化提取出的圆环区域
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(ring_region, cv2.COLOR_BGR2RGB))
plt.title("提取的圆环区域")
plt.axis("off")
plt.show()
# 转换为 HSV 空间
hsv = cv2.cvtColor(ring_region, cv2.COLOR_BGR2HSV)

# 提取圆环区域内的像素
masked_pixels = hsv[mask == 255]

# 判断主色
if len(masked_pixels) > 0:
    mean_hue = np.mean(masked_pixels[:, 0])
    result = detect_red_or_blue(mean_hue)
    print(f"平均色调 H = {mean_hue:.2f} → 分类结果：{result}")
else:
    print("⚠️ 圆环区域内没有有效像素，无法判断颜色")

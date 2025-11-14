import time

import cv2
import numpy as np

class ColorDetector:
    def __init__(self):
        pass

    def detect_color(self, image: np.ndarray) -> str:
        """
        输入：RGB图像（BGR 格式，OpenCV 默认格式）
        输出：'红色'、'蓝色' 或 '未知'
        """
        start = time.time()
        # Step 1: 灰度 + 增强
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        edges = cv2.Canny(enhanced, 100, 200)

        # Step 2: 检测圆（Hough）
        height, width = edges.shape
        img_center = (width // 2, height // 2)
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
            return "未知"

        # Step 3: 找最中心的圆
        circles = np.uint16(np.around(best_circles))
        best_circle = None
        best_score = float('inf')

        for circle in circles[0, :]:
            cx, cy, r = circle
            dist = np.hypot(cx - img_center[0], cy - img_center[1])
            if dist < best_score:
                best_score = dist
                best_circle = circle

        if best_circle is None:
            return "未知"

        # Step 4: 提取圆环区域并转换到HSV
        cx, cy, r = best_circle
        r_outer = int(r)
        r_inner = int(r * 0.85)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r_outer, 255, -1)
        cv2.circle(mask, (cx, cy), r_inner, 0, -1)
        ring_region = cv2.bitwise_and(image, image, mask=mask)
        hsv = cv2.cvtColor(ring_region, cv2.COLOR_BGR2HSV)

        # Step 5: 平均色调判断
        masked_pixels = hsv[mask == 255]
        if len(masked_pixels) == 0:
            return "未知"

        mean_hue = np.mean(masked_pixels[:, 0])
        color_result = self._detect_red_or_blue(mean_hue)
        end = time.time()
        print(f"执行耗时：{end - start:.2f} 秒")
        return color_result, ring_region

    def _detect_red_or_blue(self, hue: float) -> str:
        hue = int(hue)
        if hue <= 70 or hue >= 140:
            return "红色"
        elif 70 <= hue <= 140:
            return "蓝色"
        else:
            return "未知"


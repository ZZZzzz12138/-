import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import re

from model import ProtoNetWithAngle

# ==== 配置 ====
N_WAY = 3
K_SHOT = 3
IMAGE_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载模型 ====
model = ProtoNetWithAngle(n_way=N_WAY).to(device)
model.load_state_dict(torch.load("proto_angle_final.pt", map_location=device))
model.eval()

# ==== 支持集构造 ====
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor()
])

class_names = sorted(os.listdir("raw"))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

support_imgs, support_labels = [], []
for cls in class_names:
    img_paths = glob.glob(os.path.join("raw", cls, "*.jpg"))
    selected = img_paths[:K_SHOT]
    for path in selected:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        support_imgs.append(img)
        support_labels.append(class_to_idx[cls])

support_imgs = torch.stack(support_imgs).to(device)
support_labels = torch.tensor(support_labels).to(device)

# ==== 测试集预测 ====

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# 替换 sorted() 语句
test_paths = sorted(glob.glob("test_images/*.*"), key=natural_key)
to_pil = T.ToPILImage()

# Model Manager Module
# 模型管理器模块

import cv2
import numpy as np
import time

from .circle_detector import CircleDetector
from .color_detector import ColorDetector
from .label_recognizer import LabelRecognizer
from .angle_estimator import AngleEstimator


class ModelManager:
    """
    模型管理器类，作为图像识别模块的统一入口

    功能：协调各个子模块，完成完整的图像识别流程
    """

    def __init__(self, tesseract_path=None):
        """
        初始化模型管理器

        参数:
            tesseract_path (str, optional): Tesseract OCR引擎的路径
        """
        # 初始化各个子模块
        self.circle_detector = CircleDetector()
        self.color_detector = ColorDetector()
        self.label_recognizer = LabelRecognizer(tesseract_path)
        self.angle_estimator = AngleEstimator()

        # 最近一次的识别结果
        self.last_result = None

        # 处理帧计数和FPS计算
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def run(self, image):
        """
        运行完整的图像识别流程

        参数:
            image (numpy.ndarray): 输入图像

        返回:
            dict: 识别结果，格式如下：
                {
                    "valid": bool,  # 是否成功识别
                    "circle": {     # 若valid为True，则包含圆形信息
                        "center": [x, y],  # 圆心坐标
                        "radius": r        # 圆半径
                    },
                    "color": str,   # 颜色名称
                    "label": str,   # 标签文本
                    "angle": float  # 旋转角度
                }
        """
        # 更新帧计数和FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:  # 每秒更新一次FPS
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        # 初始化结果字典
        result = {"valid": False}

        # 1. 圆形检测
        circle_result = self.circle_detector.detect(image)

        # 如果未检测到圆形，返回无效结果
        if not circle_result["valid"]:
            self.last_result = result
            return result

        # 更新结果
        result.update(circle_result)

        # 提取圆形ROI
        roi, roi_pos = self.circle_detector.extract_roi(image, circle_result["circle"])

        # 2. 颜色识别
        color = self.color_detector.detect_color(roi)
        result["color"] = color

        # 3. 标签识别
        label = self.label_recognizer.recognize(roi)
        result["label"] = label

        # 4. 角度估算
        angle, center, dimensions = self.angle_estimator.estimate(roi)
        result["angle"] = angle

        # 保存最近一次的识别结果
        self.last_result = result

        return result

    def get_overlay(self, image):
        """
        在图像上叠加显示识别结果

        参数:
            image (numpy.ndarray): 输入图像

        返回:
            numpy.ndarray: 叠加了识别结果的图像
        """
        # 复制原图，避免修改原始数据
        overlay = image.copy()

        # 如果没有有效的识别结果，只显示FPS
        if self.last_result is None or not self.last_result["valid"]:
            # 添加FPS信息
            cv2.putText(
                overlay,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            # 添加状态信息
            cv2.putText(
                overlay,
                "Status: Waiting for object",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            return overlay

        # 获取最近一次的识别结果
        result = self.last_result

        # 1. 绘制圆形检测结果
        if "circle" in result:
            overlay = self.circle_detector.draw_result(overlay, result["circle"])

        # 2. 绘制颜色识别结果
        if "color" in result:
            overlay = self.color_detector.draw_result(overlay, result["color"])

        # 3. 绘制标签识别结果
        if "label" in result:
            overlay = self.label_recognizer.draw_result(overlay, result["label"])

        # 4. 绘制角度估算结果
        if "angle" in result:
            # 获取圆心和半径
            center = result["circle"]["center"]
            radius = result["circle"]["radius"]

            # 估算旋转中心在原图中的位置
            rot_center = (center[0], center[1])

            # 绘制角度
            overlay = self.angle_estimator.draw_result(
                overlay,
                result["angle"],
                rot_center,
                (radius * 2, radius * 2)
            )

        # 添加FPS信息
        cv2.putText(
            overlay,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 添加状态信息
        status = "Status: Object detected"
        color = (0, 255, 0)  # 绿色表示成功

        cv2.putText(
            overlay,
            status,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return overlay

    @staticmethod
    def format_result(result):
        """
        格式化识别结果为JSON格式

        参数:
            result (dict): 识别结果

        返回:
            dict: 格式化后的结果
        """
        # 如果结果无效，返回简单结果
        if not result["valid"]:
            return {"valid": False}

        # 格式化为文档要求的JSON结构
        formatted = {
            "valid": True,
            "circle": {
                "center": result["circle"]["center"],
                "radius": result["circle"]["radius"]
            },
            "color": result["color"],
            "label": result["label"],
            "angle": result["angle"]
        }

        return formatted

    # ✅ 可视化：左边为原图，右边为模型实际输入
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig)
    plt.title("original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(to_pil(img_tensor_raw.cpu()))
    plt.title(f"put\\n{pred_class_name}, {pred_angle:.1f}°")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
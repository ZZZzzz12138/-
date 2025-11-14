import math
from typing import Dict

import cv2
import numpy as np

from backend.color_detector import ColorDetector
from backend.pattern_classifier.pattern_angle_predictor import PatternAndAngleClassifier


class VisionRecognitionService:
    def __init__(self,
                 angle_model_path: str = "proto_angle_final.pt",
                 enable_color: bool = True,
                 enable_pattern: bool = True,
                 enable_angle: bool = True):
        """
        初始化视觉识别服务
        - 模型只用来做“类别识别”
        - 角度预测使用：特征匹配 + 部分仿射矩阵 -> 旋转角
        """
        self.color_enabled = enable_color
        self.pattern_enabled = enable_pattern
        self.angle_enabled = enable_angle

        if self.color_enabled:
            self.color_detector = ColorDetector()

        if self.pattern_enabled:
            # 仍然复用原来的分类器，但只取 pattern，不用它的 angle
            self.pattern_classifier = PatternAndAngleClassifier(
                model_path=angle_model_path
            )

        # 角度估计用到的模板
        if self.angle_enabled:
            self._init_angle_templates()

    # ===== 角度模板相关 =====
    def _init_angle_templates(self):
        """
        为每个图案类别加载一张“0°姿态”的模板图（灰度图）
        按你的真实类别名和模板路径修改 pattern2path 即可
        """
        self.angle_templates: Dict[str, np.ndarray] = {}

        # TODO: 按实际类别名和模板路径修改
        pattern2path = {
            "PATTERN_A": r"D:\Desktop\all\SmartAssemblyVisionApp\backend\raw\class1\img01.jpg",
            "PATTERN_B": r"D:\Desktop\all\SmartAssemblyVisionApp\backend\raw\class2\img01.jpg",
            "PATTERN_C": r"D:\Desktop\all\SmartAssemblyVisionApp\backend\raw\class3\img01.jpg",
        }

        for pattern, path in pattern2path.items():
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                self.angle_templates[pattern] = tmpl
            else:
                print(f"[AngleTemplate] 模板加载失败: {pattern} -> {path}")

        # 默认模板（当找不到对应 pattern 时使用）
        self.default_angle_template = None
        if self.angle_templates:
            self.default_angle_template = next(iter(self.angle_templates.values()))

    # ===== 辅助函数 =====
    @staticmethod
    def _center_crop_square(gray: np.ndarray, target_size: int = 512) -> np.ndarray:
        """
        中心裁剪为正方形并缩放到 target_size，用于模板 & ROI 统一尺寸
        """
        h, w = gray.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = gray[y0:y0 + side, x0:x0 + side]
        if crop.shape[0] != target_size:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return crop

    def _detect_and_match(self, img1: np.ndarray, img2: np.ndarray):
        """
        提取 img1/img2 特征并做匹配
        返回：kps1, kps2, good_matches
        """
        # 1. 选择特征算子
        if hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
            method_name = "SIFT"
        else:
            detector = cv2.ORB_create(2000)
            norm_type = cv2.NORM_HAMMING
            method_name = "ORB"

        # 2. 提取特征
        kps1, des1 = detector.detectAndCompute(img1, None)
        kps2, des2 = detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kps1) < 4 or len(kps2) < 4:
            print(f"[ANGLE] {method_name} 特征不足：tmpl_kps={len(kps1)}, roi_kps={len(kps2)}")
            return None, None, []

        # 3. 匹配 + Lowe ratio test
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print(f"[ANGLE] {method_name} good matches:", len(good))
        return kps1, kps2, good

    @staticmethod
    def _calculate_rotation_from_affine(affine_matrix: np.ndarray):
        """
        从 2x3 部分仿射矩阵中计算旋转角度
        参数:
            affine_matrix: cv2.estimateAffinePartial2D 返回的 2x3 矩阵
        返回:
            旋转角度（度），失败返回 None
        """
        if affine_matrix is None or affine_matrix.shape != (2, 3):
            print("[ANGLE] 错误：仿射矩阵无效")
            return None

        # 线性部分 2x2
        a00, a01 = affine_matrix[0, 0], affine_matrix[0, 1]
        a10, a11 = affine_matrix[1, 0], affine_matrix[1, 1]

        # 对于 “旋转 + 等比缩放” 的情况：
        # [ a00 a01 ] ≈ s [ cosθ  -sinθ ]
        # [ a10 a11 ]   s [ sinθ   cosθ ]
        # 旋转角度可以直接用 atan2(a10, a00)
        theta_rad = math.atan2(a10, a00)
        theta_deg = math.degrees(theta_rad)

        # 规范到 [-180, 180]
        if theta_deg <= -180.0:
            theta_deg += 360.0
        if theta_deg > 180.0:
            theta_deg -= 360.0

        return theta_deg

    def _estimate_affine(self, img_tmpl: np.ndarray, img_roi: np.ndarray):
        """
        使用特征匹配 + RANSAC 估计 模板 -> ROI 的部分仿射矩阵
        """
        kps1, kps2, good = self._detect_and_match(img_tmpl, img_roi)
        if not good or len(good) < 4 or kps1 is None or kps2 is None:
            print("[ANGLE] 有效匹配点不足，无法估计仿射矩阵")
            return None

        src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC 拟合部分仿射变换（只允许旋转+缩放+平移）
        affine, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )

        inlier_count = int(inliers.sum()) if inliers is not None else 0
        print("[ANGLE] Affine 估计完成，内点数:", inlier_count)

        if affine is None or inlier_count < 4:
            print("[ANGLE] 仿射矩阵估计失败或内点过少")
            return None

        return affine

    # ===== 自定义角度任务（特征匹配 + 部分仿射矩阵） =====
    def _custom_angle_task(self, image, result_dict) -> float:
        if not self.angle_enabled:
            return 0.0

        # 1. 模板选择
        pattern = result_dict.get("pattern", "")
        print("[ANGLE] pattern from classifier:", pattern)

        tmpl = self.angle_templates.get(pattern, self.default_angle_template)
        if tmpl is None or tmpl.size == 0:
            print("[ANGLE] NO VALID TEMPLATE for pattern:", pattern)
            return 0.0
        else:
            print("[ANGLE] template shape:", tmpl.shape)

        # 2. 预处理当前图像：转灰度 + 中心裁剪
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 默认认为圆形徽章在画面中心，这里做一个中心方形裁剪
        roi = self._center_crop_square(gray, target_size=512)
        tmpl_proc = self._center_crop_square(tmpl, target_size=512)

        # （可选）轻微模糊，稳定特征
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        tmpl_proc = cv2.GaussianBlur(tmpl_proc, (3, 3), 0)

        # 3. 估计模板 -> ROI 的部分仿射矩阵
        affine = self._estimate_affine(tmpl_proc, roi)
        if affine is None:
            return 0.0

        # 4. 从仿射矩阵中提取旋转角
        theta_deg = self._calculate_rotation_from_affine(affine)
        if theta_deg is None:
            return 0.0

        # 这里 theta_deg 表示：模板坐标系相对于 ROI 的旋转角度
        # 一般我们把“物体本身的旋转”记为反向，所以取负号
        angle_obj = -float(theta_deg)

        # 归一化到 [-180, 180]
        while angle_obj <= -180.0:
            angle_obj += 360.0
        while angle_obj > 180.0:
            angle_obj -= 360.0

        print(f"[ANGLE] theta_deg(from Affine): {theta_deg:.2f}, final angle: {angle_obj:.2f}")
        return angle_obj

    # ===== 总入口 =====
    def predict_all(self, image) -> Dict:
        """
        输入：OpenCV格式的图像（BGR）
        输出：包含颜色识别、图案识别、角度预测的结果字典
        """
        result: Dict = {}

        # 1. 颜色识别
        if self.color_enabled:
            try:
                result["color"] = self.color_detector.detect_color(image)
            except Exception as e:
                result["color"] = "识别错误"
                result["color_error"] = str(e)

        # 2. 图案类别识别（只取 pattern，不用模型的 angle）
        if self.pattern_enabled:
            try:
                pattern, _ = self.pattern_classifier.predict(image)
                result["pattern"] = pattern
            except Exception as e:
                result["pattern"] = "识别错误"
                result["pattern_error"] = str(e)

        # 3. 角度预测（特征匹配 + 部分仿射矩阵）
        if self.angle_enabled:
            try:
                angle = self._custom_angle_task(image, result)
                result["angle"] = angle
            except Exception as e:
                result["angle"] = -1
                result["angle_error"] = str(e)

        return result

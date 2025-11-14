# recognition_controller.py
import os


class RecognitionController:
    """
    负责把一帧图像丢给 VisionRecognitionService，并整理结果。
    不操作 UI，只返回纯数据。
    """

    def __init__(self, recognizer):
        """
        recognizer: 传入 backend.vision_service.VisionRecognitionService 的实例
        """
        self.recognizer = recognizer

    def run_inference(self, frame_bgr):
        """
        传入一帧 BGR 图像，返回 dict:
        {
            "success": True/False,
            "error": str 或 None,
            "color": str,
            "pattern": str,
            "angle": float,
            "color_image": np.ndarray 或 None,
            "pattern_img_path": str 或 None,
        }
        """
        if frame_bgr is None:
            return {
                "success": False,
                "error": "推理图像为空",
            }

        try:
            result = self.recognizer.predict_all(frame_bgr)

            color, color_image = result.get("color", ("未知", None))
            pattern = result.get("pattern", "")
            angle = result.get("angle", 0)

            # 角度标准化
            if isinstance(angle, (int, float)):
                angle_val = float(angle)
            else:
                try:
                    angle_val = float(angle)
                except Exception:
                    angle_val = 0.0

            # 分类过程图路径（可选）
            pattern_img_path = None
            if pattern:
                pattern_img_path = os.path.join(
                    "backend", "raw", pattern, "img01.jpg"
                )

            return {
                "success": True,
                "error": None,
                "color": color,
                "pattern": pattern,
                "angle": angle_val,
                "color_image": color_image,
                "pattern_img_path": pattern_img_path,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

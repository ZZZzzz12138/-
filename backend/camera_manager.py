# camera_manager.py
import cv2
from PyQt5.QtCore import QObject, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


class CameraManager(QObject):
    """
    负责摄像头打开、预览、定格，以及过程图显示。
    UI 文本（label）通过构造函数传进来。
    """

    def __init__(
        self,
        camera_label,
        process_label1,
        process_label2,
        process_info1,
        process_info2,
        result_label,
        status_label,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_label = camera_label
        self.process_label1 = process_label1
        self.process_label2 = process_label2
        self.process_info1 = process_info1
        self.process_info2 = process_info2
        self.result_label = result_label
        self.status_label = status_label

        self.cap = None
        self.preview_running = False
        self.frozen_frame = None
        self._last_frame_bgr = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    # ========= 预览相关 =========
    def start_preview(self):
        """打开相机，进入实时预览（不做推理）"""
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.result_label.setText("❌ 无法访问摄像头设备")
            self.status_label.setText("状态: 错误 - 摄像头连接失败")
            return
        try:
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.preview_running = True
        self.frozen_frame = None
        self.timer.start(0)
        self.result_label.setText("👀 实时预览中（未开始识别）")
        self.status_label.setText("状态: 摄像头已打开 | 预览运行中")

    def resume_preview(self):
        """从定格/停顿回到实时预览"""
        if self.cap is None or not self.cap.isOpened():
            self.start_preview()
            return
        self.preview_running = True
        self.frozen_frame = None
        if not self.timer.isActive():
            self.timer.start(0)

    def stop_camera(self):
        """停止摄像头和计时器"""
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview_running = False
        self.frozen_frame = None
        self.camera_label.setText("📡 视觉系统已停止")

    # ========= 定格 & 帧获取 =========
    def freeze_current_frame(self):
        """
        定格当前画面，停止预览，返回这一帧 BGR 图像。
        如果没有帧则返回 None。
        """
        if self._last_frame_bgr is None:
            return None
        self.frozen_frame = self._last_frame_bgr.copy()
        self.preview_running = False
        if self.timer.isActive():
            self.timer.stop()

        self._show_bgr_on_label(self.frozen_frame, self.camera_label)
        return self.frozen_frame

    def get_last_frame(self):
        """如果需要，只读当前缓存帧（不停止预览）"""
        return self._last_frame_bgr

    # ========= 计时器刷新帧 =========
    def update_frame(self):
        """预览阶段：只读帧并显示；识别阶段：不读帧（维持定格）"""
        if self.cap is None:
            return

        if not self.preview_running:
            if self.frozen_frame is not None:
                self._show_bgr_on_label(self.frozen_frame, self.camera_label)
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_label.setText("状态: 读取视频帧失败…")
            return

        # 缓存当前帧给“定格并推理”用
        self._last_frame_bgr = frame
        self._show_bgr_on_label(frame, self.camera_label)

        # 预览提示
        if "预览" not in self.result_label.text():
            self.result_label.setText(
                "👀 实时预览中（点击“启动识别”将定格并开始推理）"
            )

    # ========= 图像显示辅助 =========
    def _show_bgr_on_label(self, bgr_image, label):
        """把 BGR 图像显示到指定 QLabel"""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)
        label.setText("")

    def update_process_image(self, label, image, title=None):
        """在标签中显示处理后的图像，并更新说明文本"""
        if image is None:
            return

        if len(image.shape) == 2:
            # 灰度图
            h, w = image.shape
            qt_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # BGR 转 RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(
                rgb_image.data,
                w,
                h,
                ch * w,
                QImage.Format_RGB888,
            )

        pixmap = QPixmap.fromImage(qt_image).scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

        # 更新信息文本
        if title:
            info_text = (
                f"Pattern: {title}"
                if label == self.process_label2
                else f"Color: {title}"
            )
            if label == self.process_label2:
                self.process_info2.setText(info_text)
            else:
                self.process_info1.setText(info_text)

    # ========= 清理 =========
    def cleanup(self):
        """关闭摄像头和计时器（程序退出时调用）"""
        try:
            self.stop_camera()
        except Exception:
            pass

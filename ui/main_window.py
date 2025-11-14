# main_window.py
import os
import sys
import time

import cv2
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

from backend.camera_manager import CameraManager
from backend.hardware_controller import HardwareController
from backend.recognition_controller import RecognitionController
from backend.vision_service import VisionRecognitionService

from ui.ui_main_window import MainWindowUI


# ==== 串口相关配置 ====
ENABLE_SERIAL = False  # 想临时彻底禁用串口，改成 False
END = b"\xff\xff\xff"


class MainWindow(QWidget, MainWindowUI):
    def __init__(self):
        super().__init__()

        # ===== 窗口与样式 =====
        self.setWindowTitle("智能翻转机构视觉识别系统")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0A0A15, stop:0.5 #050508, stop:1 #000000);
                color: #E8E8FF;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            }

            QLabel#title {
                font-size: 36px;
                font-weight: 300;
                color: #FFFFFF;
                letter-spacing: 2px;
                background: none;
                border: none;
            }

            QLabel#result_label {
                font-size: 20px;
                font-weight: 400;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 240, 255, 0.1),
                    stop:0.5 rgba(0, 120, 180, 0.15),
                    stop:1 rgba(0, 60, 120, 0.1));
                border: 2px solid rgba(0, 240, 255, 0.3);
                border-radius: 15px;
                padding: 20px;
                min-height: 60px;
            }

            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 180, 240, 0.8),
                    stop:1 rgba(0, 120, 180, 0.6));
                color: white;
                border: 2px solid rgba(0, 240, 255, 0.5);
                border-radius: 12px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 500;
                min-width: 140px;
                min-height: 50px;
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 220, 255, 0.9),
                    stop:1 rgba(0, 160, 220, 0.7));
                border-color: rgba(0, 255, 255, 0.8);
            }

            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 140, 200, 0.7),
                    stop:1 rgba(0, 100, 150, 0.5));
            }

            QPushButton#start_button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 200, 100, 0.8),
                    stop:1 rgba(0, 150, 80, 0.6));
                border-color: rgba(0, 255, 150, 0.5);
            }

            QPushButton#start_button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 240, 120, 0.9),
                    stop:1 rgba(0, 180, 100, 0.7));
                border-color: rgba(0, 255, 180, 0.8);
            }

            QPushButton#stop_button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 100, 100, 0.8),
                    stop:1 rgba(200, 60, 60, 0.6));
                border-color: rgba(255, 150, 150, 0.5);
            }

            QPushButton#stop_button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 140, 140, 0.9),
                    stop:1 rgba(220, 80, 80, 0.7));
                border-color: rgba(255, 180, 180, 0.8);
            }

            QGroupBox {
                border: 2px solid rgba(0, 240, 255, 0.3);
                border-radius: 15px;
                margin-top: 25px;
                font-size: 18px;
                font-weight: 400;
                color: #E8E8FF;
                padding: 15px;
                background: rgba(0, 50, 80, 0.1);
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 15px;
                background: rgba(0, 0, 0, 0.8);
                border-radius: 8px;
            }

            QLabel#process_label {
                min-height: 220px;
                background: rgba(0, 20, 40, 0.3);
                border: 1px solid rgba(0, 180, 220, 0.3);
                border-radius: 10px;
            }

            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 240, 255, 0.3),
                    stop:0.5 rgba(0, 180, 220, 0.5),
                    stop:1 rgba(0, 240, 255, 0.3));
                width: 3px;
                border-radius: 1px;
            }
        """
        )

        # ===== 识别服务 =====
        print("[DEBUG] Step 2: 准备初始化 VisionRecognitionService")
        self.recognizer = VisionRecognitionService(
            angle_model_path=r"proto_angle_final.pt"
        )
        print("[DEBUG] Step 3: VisionRecognitionService 初始化完成")

        # ===== 创建 UI =====
        self.setup_ui()
        print("[DEBUG] Step 5: setup_ui 完成")

        # ===== 摄像头管理 =====
        self.camera_manager = CameraManager(
            camera_label=self.camera_label,
            process_label1=self.process_label1,
            process_label2=self.process_label2,
            process_info1=self.process_info1,
            process_info2=self.process_info2,
            result_label=self.result_label,
            status_label=self.status_label,
            parent=self,
        )
        print("[DEBUG] Step 6: CameraManager 初始化完成")

        # ===== 识别控制器 =====
        self.recognition_controller = RecognitionController(self.recognizer)
        print("[DEBUG] Step 7: RecognitionController 初始化完成")

        # ===== 硬件 / 串口控制 =====
        self.hardware_controller = HardwareController(
            enable_serial=ENABLE_SERIAL,
            log_func=self.add_serial_log,
            status_label=self.status_label,
            parent=self,
        )

        self.hardware_controller.setup_serial(
            trigger_callback=self.trigger_recognition,
            command_callback=self.handle_hmi_command,  # 新增
        )
        print("[DEBUG] Step 8: HardwareController 初始化完成")

        # ===== 信号连接 =====
        self.setup_connections()
        print("[DEBUG] Step 9: setup_connections 完成")

        # ===== 自动启动预览 =====
        print("[DEBUG] Step 11: 准备启动自动预览")
        QTimer.singleShot(0, self.camera_manager.start_preview)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        print("[DEBUG] Step 12: MainWindow.__init__ 结束")


    def handle_hmi_command(self, cmd: str):
        """
        串口屏发来的设备控制指令。
        注意：这里既可以直接调 hardware_controller 的方法，
             也可以通过 .click() 模拟点按钮，让界面有反馈。
        """

        self.add_serial_log(f"📟 串口屏命令: {cmd}")

        if cmd == "STEPPER_FWD":
            # 方案1：直接调用动作（更干净）
            self.hardware_controller.stepper_forward()
            # 方案2：模拟点按钮，让按钮高亮/动画
            # self.stepper_forward_btn.click()

        elif cmd == "STEPPER_BWD":
            self.hardware_controller.stepper_backward()
            # self.stepper_backward_btn.click()

        elif cmd == "STEPPER_STOP":
            self.hardware_controller.stepper_stop()
            # self.stepper_stop_btn.click()

        elif cmd == "FLIP_FORWARD":
            self.hardware_controller.flip_forward()
            # self.flip_forward_btn.click()

        elif cmd == "FLIP_BACKWARD":
            self.hardware_controller.flip_backward()
            # self.flip_backward_btn.click()

        elif cmd == "FLIP_BASE":
            self.hardware_controller.flip_stop()
            # self.flip_stop_btn.click()
        elif cmd == "DEVICE_CTRL":
            # 比如切到“设备控制”这个 tab
            # 假设第 2 个 tab 是设备控制（从 0 开始数）
            self.tab_widget.setCurrentIndex(1)
        else:
            self.add_serial_log(f"⚠️ 未知 HMI 命令: {cmd}")




    # ========= 信号连接 =========
    def setup_connections(self):
        # 识别控制按钮
        self.start_button.clicked.connect(self.freeze_and_infer)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.quit_button.clicked.connect(self.close)

        # 步进电机控制
        self.stepper_forward_btn.clicked.connect(
            self.hardware_controller.stepper_forward
        )
        self.stepper_backward_btn.clicked.connect(
            self.hardware_controller.stepper_backward
        )
        self.stepper_stop_btn.clicked.connect(
            self.hardware_controller.stepper_stop
        )

        # 翻转机构控制
        self.flip_forward_btn.clicked.connect(
            self.hardware_controller.flip_forward
        )
        self.flip_backward_btn.clicked.connect(
            self.hardware_controller.flip_backward
        )
        self.flip_stop_btn.clicked.connect(self.hardware_controller.flip_stop)

        # 日志控制
        self.clear_log_btn.clicked.connect(self.clear_serial_log)

    # ========= 识别流程 =========
    def freeze_and_infer(self):
        """定格当前画面，并用这一帧做推理，然后把结果显示+回传串口"""
        frame = self.camera_manager.freeze_current_frame()
        if frame is None:
            self.status_label.setText("状态: 尚未获取到视频帧，无法定格")
            return

        self.result_label.setText("🧊 画面已定格，开始识别…")
        self.status_label.setText("状态: 推理中…")
        self.start_button.setEnabled(False)

        res = self.recognition_controller.run_inference(frame)
        if not res.get("success"):
            self.result_label.setText(f"❌ 识别失败：{res.get('error')}")
            self.status_label.setText("状态: 推理异常")
            self.stop_button.setEnabled(True)
            return

        color = res.get("color", "未知")
        pattern = res.get("pattern", "")
        angle_val = res.get("angle", 0.0)
        color_image = res.get("color_image", None)
        pattern_img_path = res.get("pattern_img_path", None)

        display_text = f"""
        <span style='font-size:18px; color:#00E0FF; font-weight:500;'>🎯 AI识别结果&nbsp;&nbsp;</span>
        <span style='color:#80D0FF;'>颜色特征：</span><span style='color:#FFFFFF;'>{color}</span>&nbsp;&nbsp;
        <span style='color:#80D0FF;'>模式分类：</span><span style='color:#FFFFFF;'>{pattern}</span>&nbsp;&nbsp;
        <span style='color:#80D0FF;'>角度检测：</span><span style='color:#FFFFFF;'>{angle_val:.1f}°</span>
        """
        self.result_label.setText(display_text)

        # 过程图（分类）
        if pattern:
            if pattern_img_path and os.path.exists(pattern_img_path):
                pattern_img = cv2.imread(pattern_img_path)
                if pattern_img is not None:
                    self.camera_manager.update_process_image(
                        self.process_label2, pattern_img, pattern
                    )
                else:
                    self.process_label2.setText("⚠️ 分类图像读取失败")
                    self.process_info2.setText(pattern or "Unknown")
            else:
                self.process_label2.setText("⚠️ 分类图像未找到")
                self.process_info2.setText(pattern or "Unknown")

        # 过程图（颜色）
        if color_image is not None:
            self.camera_manager.update_process_image(
                self.process_label1, color_image, color
            )

        # 回传串口屏
        sent_ok = self.hardware_controller.send_results(
            color, pattern, angle_val
        )
        if sent_ok is True:
            self.status_label.setText("状态: 推理完成（已回传到串口屏）")
        elif sent_ok is False:
            self.status_label.setText(
                "状态: 推理完成（回传串口屏失败，请检查串口/端口占用）"
            )
        else:
            # 串口未启用的情况，保持当前状态提示即可
            pass

        self.stop_button.setEnabled(True)

    def on_stop_clicked(self):
        """停止识别，恢复到实时预览"""
        self.camera_manager.resume_preview()
        self.result_label.setText("👀 已恢复实时预览")
        self.status_label.setText("状态: 预览运行中")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    # ========= 串口日志 =========
    def add_serial_log(self, message: str):
        """添加串口通信日志"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # 在 UI 线程中安全更新
        QTimer.singleShot(0, lambda: self._update_log_text(log_entry))

    def _update_log_text(self, log_entry: str):
        """线程安全地更新日志文本"""
        self.serial_log_text.append(log_entry)
        # 自动滚动到底部
        self.serial_log_text.verticalScrollBar().setValue(
            self.serial_log_text.verticalScrollBar().maximum()
        )

    def clear_serial_log(self):
        """清空串口日志"""
        self.serial_log_text.clear()
        self.add_serial_log("日志已清空")

    # ========= 串口启动信号回调 =========
    def trigger_recognition(self):
        """接到串口屏启动信号时的处理：提示→启动识别"""
        self.result_label.setText("🚀 收到启动信号，开始识别...")
        self.freeze_and_infer()

    # ========= 关闭 & 清理 =========
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            self.camera_manager.cleanup()
        except Exception:
            pass

        try:
            self.hardware_controller.shutdown()
        except Exception:
            pass

        event.accept()



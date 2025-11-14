# hardware_controller.py
from PyQt5.QtCore import QObject

from backend.serial_io import ScreenSender, SerialBridge


class HardwareController(QObject):
    """
    负责串口初始化、串口桥接线程、以及步进电机/翻转机构控制。
    UI 更新通过回调函数传入（log_func、status_label、trigger_callback）。
    """

    def __init__(
        self,
        enable_serial: bool,
        log_func=None,
        status_label=None,
        parent=None,
    ):
        super().__init__(parent)
        self.enable_serial = enable_serial
        self.log = log_func or (lambda msg: None)
        self.status_label = status_label

        self.screen_sender = None
        self.device_sender = None
        self.serial_bridge = None

        # 串口回调引用（启动识别 / 设备控制）
        self._trigger_callback = None
        self._command_callback = None

    # ========= 串口初始化 =========
    def setup_serial(self, trigger_callback=None, command_callback=None):
        """根据 enable_serial 初始化串口和监听线程"""
        # 记录回调，供后续 SerialBridge 或其他模块使用
        self._trigger_callback = trigger_callback
        self._command_callback = command_callback

        if not self.enable_serial:
            print("[SerialInit] 串口功能已禁用（ENABLE_SERIAL=False）")
            if self.status_label:
                self.status_label.setText(
                    "状态: 串口功能已禁用，仅使用本地按钮控制"
                )
            self.log("串口功能已禁用，仅使用本地按钮控制")
            return

        print("[DEBUG] Step 9: 准备初始化串口（如果启用）")
        try:
            # 1) 串口屏：COM3
            self.screen_sender = ScreenSender(
                port="COM3",
                baud=115200,
                encoding="gbk",
                log_func=self.log,  # 把同一套日志函数传给 ScreenSender
            )

            opened = self.screen_sender.open()
            if opened:
                if self.status_label:
                    self.status_label.setText(
                        "状态: 串口已连接 | 等待外部启动信号"
                    )
                self.log("串口已连接，等待外部启动信号")
            else:
                if self.status_label:
                    self.status_label.setText(
                        "状态: 串口未连接，将仅使用本地按钮控制"
                    )
                self.log("串口未连接，将仅使用本地按钮控制")

            # 2) 下位机/电机控制串口：COM5
            self.device_sender = ScreenSender(
                port="COM11",
                baud=9600,
                encoding="gbk",
                log_func=self.log,
                init_nextion=False,
            )
            device_opened = self.device_sender.open()
            if device_opened:
                self.log("下位机串口已连接 (COM5)，可以发送步进/翻转控制指令")
            else:
                self.log("下位机串口 (COM5) 打开失败，将无法通过 COM5 控制电机")

            # 3) 串口监听线程：监听 COM3（串口屏）
            self.serial_bridge = SerialBridge(
                port=self.screen_sender.port if self.screen_sender else None,
                baud=self.screen_sender.baud
                if self.screen_sender
                else 115200,
                ser=self.screen_sender.ser if self.screen_sender else None,
            )

            # 串口桥接线程发出的“启动识别”信号
            if self._trigger_callback and hasattr(
                self.serial_bridge, "start_signal"
            ):
                self.serial_bridge.start_signal.connect(self._trigger_callback)

            # 串口桥接线程发出的“设备控制命令”信号（需要在 SerialBridge 中定义 command_signal: pyqtSignal(str)）
            if self._command_callback and hasattr(
                self.serial_bridge, "command_signal"
            ):
                self.serial_bridge.command_signal.connect(self._command_callback)

            # 串口日志 → 状态栏 / 日志窗口
            def _on_log(msg: str):
                if self.status_label:
                    self.status_label.setText(f"状态: {msg}")
                self.log(msg)

            self.serial_bridge.log_signal.connect(_on_log)
            self.serial_bridge.start()

        except Exception as e:
            print(f"[SerialInit] 串口初始化异常（不会中止程序）: {e}")
            if self.status_label:
                self.status_label.setText(
                    "状态: 串口初始化异常，仅使用本地按钮控制"
                )
            self.log(f"串口初始化异常，仅使用本地按钮控制：{e}")

        print("[DEBUG] Step 10: 串口初始化逻辑执行完毕")

    # ========= 推理结果发送 =========
    def send_results(self, color: str, pattern: str, angle: float):
        """
        把识别结果通过串口推送出去。
        返回:
          True  -> 发送成功
          False -> 发送失败
          None  -> 串口未启用或未初始化
        """
        if not self.enable_serial or not self.screen_sender:
            return None

        try:
            sent_ok = self.screen_sender.push_results(color, pattern, angle)
            if sent_ok:
                self.log("识别结果已回传到串口屏")
                return True
            else:
                self.log("识别结果回传串口屏失败")
                return False
        except Exception as e:
            self.log(f"发送识别结果失败: {e}")
            return False

    # ========= 步进电机控制 =========
    def _send_command(self, desc: str, cmd: str):
        """通过串口把控制指令发给下位机（优先使用 COM5）"""
        self.log(f"发送指令: {desc}")
        try:
            sender = getattr(self, "device_sender", None) or getattr(
                self, "screen_sender", None
            )
            if sender:
                ok = sender.send_command(cmd)
                if ok:
                    self.log(f"指令已发送: {cmd}")
                else:
                    self.log(f"指令发送失败: {cmd}")
            else:
                self.log("发送失败：未初始化任何串口发送器")
        except Exception as e:
            self.log(f"发送失败: {str(e)}")

    def stepper_forward(self):
        """步进电机正转"""
        self._send_command("步进电机正转", "STEPPER_FORWARD")

    def stepper_backward(self):
        """步进电机反转"""
        self._send_command("步进电机反转", "STEPPER_BACKWARD")

    def stepper_stop(self):
        """步进电机停止"""
        self._send_command("步进电机停止", "STEPPER_STOP")

    # ========= 翻转机构控制 =========
    def flip_forward(self):
        """正向翻转"""
        self._send_command("正向翻转", "FLIP_FORWARD")

    def flip_backward(self):
        """反向翻转"""
        self._send_command("反向翻转", "FLIP_BACKWARD")

    def flip_stop(self):
        """停止翻转"""
        self._send_command("停止翻转", "FLIP_STOP")

    # ========= 清理 =========
    def shutdown(self):
        """程序退出时调用，关闭串口线程和串口"""
        try:
            if (
                self.serial_bridge
                and self.serial_bridge.isRunning()
            ):
                self.serial_bridge.stop()
                self.serial_bridge.wait(500)
        except Exception:
            pass

        try:
            if self.screen_sender:
                self.screen_sender.close()
        except Exception:
            pass

        try:
            if (
                getattr(self, "device_sender", None)
                and self.device_sender is not self.screen_sender
            ):
                self.device_sender.close()
        except Exception:
            pass

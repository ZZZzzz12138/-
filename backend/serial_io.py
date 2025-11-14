import time
import serial
from typing import Optional, Callable

from PyQt5.QtCore import QThread, pyqtSignal

# Nextion / 串口屏帧结束符
END = b"\xff\xff\xff"


class ScreenSender:
    """
    负责 PC → 屏幕 / 下位机 的发送。
    现在增加了 log_func，每次发送也会写入串口通信日志。
    """

    def __init__(
        self,
        port: str = "COM3",
        baud: int = 115200,
        encoding: str = "gbk",
        log_func: Optional[Callable[[str], None]] = None,
        init_nextion: bool = True,
    ):
        self.port = port
        self.baud = baud
        self.encoding = encoding
        # Python 3.8 下用 Optional[...]，不要用 Serial | None
        self.ser = None  # type: Optional[serial.Serial]
        self.log = log_func or (lambda msg: None)
        self.init_nextion = init_nextion

        # 屏幕里的控件名称（按你的 HMI 工程修改）
        self.comp_color = "tColor"      # 颜色显示文本控件
        self.comp_pattern = "tPattern"  # 类别/模式显示文本控件
        self.comp_angle = "tAngle"      # 角度显示文本控件（文本方式最兼容）

    def open(self) -> bool:
        """打开串口；若已打开则直接返回 True。"""
        if self.ser and self.ser.is_open:
            self.log("串口已复用（ScreenSender 内部）")
            return True
        try:
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.1,
                write_timeout=0.2,
            )
            time.sleep(0.1)
            self.log("ScreenSender 打开串口成功: {0}".format(self.port))
            # 如用于 Nextion 串口屏，可打开回执，方便调试
            if getattr(self, "init_nextion", False):
                self.send_cmd("bkcmd=2")
            return True
        except Exception as e:
            print(f"[ScreenSender] 打开串口失败: {e}")
            self.log("ScreenSender 打开串口失败: {0}".format(e))
            self.ser = None
            return False

    def close(self):
        """关闭串口（如由本类打开）。"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.log("ScreenSender 关闭串口: {0}".format(self.port))
        except Exception:
            pass

    def send_cmd(self, text_cmd: str) -> bool:
        """
        发送纯文本命令 + FF FF FF
        例如: t0.txt=\"OK\" / page 0 / get bkcmd
        """
        if not (self.ser and self.ser.is_open):
            self.log("发送失败（串口未打开）: {0}".format(text_cmd))
            return False

        payload = text_cmd.encode(self.encoding) + END
        try:
            self.ser.write(payload)
            self.ser.flush()
            # 在日志里记录一条“电脑 → 硬件”的信息
            self.log(
                "TX CMD: {0} | RAW: {1}".format(
                    text_cmd, payload.hex(" ")
                )
            )
            return True
        except Exception as e:
            print(f"[ScreenSender] 发送失败: {e}")
            self.log("TX CMD 失败 {0}: {1}".format(text_cmd, e))
            return False

    def set_text(self, comp: str, value: str) -> bool:
        """
        设置文本控件内容。对于中文，若屏用 GBK，则 encoding 需为 'gbk'，
        屏字体也要支持对应字符。
        """
        safe = value.replace('"', '\\"')
        return self.send_cmd(f'{comp}.txt="{safe}"')

    def set_angle_text(self, comp: str, angle: float) -> bool:
        """以文本形式显示角度。"""
        return self.set_text(comp, f"{angle:.1f}°")

    def push_results(self, color: str, pattern: str, angle: float) -> bool:
        """一次性把三个识别结果写回屏幕控件。"""
        ok = True
        ok &= self.set_text(self.comp_color, color if color else "未知")
        ok &= self.set_text(self.comp_pattern, pattern if pattern else "未知")
        ok &= self.set_angle_text(
            self.comp_angle, float(angle) if angle is not None else 0.0
        )
        self.log(
            "已发送识别结果到屏幕: color={0}, pattern={1}, angle={2}".format(
                color, pattern, angle
            )
        )
        return ok

    def send_command(self, command: str) -> bool:
        """
        发送控制命令到下位机（电机/翻转机构等）。
        这里的协议你可以按实际硬件调整。
        """
        if not self.ser or not self.ser.is_open:
            self.log("发送控制命令失败（串口未打开）: {0}".format(command))
            return False

        try:
            # 根据实际硬件协议构造命令，这里示例为简单的文本 + 换行
            if command == "STEPPER_FORWARD":
                cmd_data = b"2"#b"STEP_FWD\n"
            elif command == "STEPPER_BACKWARD":
                cmd_data = b"STEP_BWD\n"
            elif command == "STEPPER_STOP":
                cmd_data = b"3"#b"STEP_STOP\n"
            elif command == "FLIP_FORWARD":
                cmd_data = b"FLIP_FWD\n"
            elif command == "FLIP_BACKWARD":
                cmd_data = b"FLIP_BWD\n"
            elif command == "FLIP_STOP":
                cmd_data = b"6"#b"FLIP_STOP\n"
            else:
                self.log("未知控制命令: {0}".format(command))
                return False

            self.ser.write(cmd_data)
            self.ser.flush()
            self.log(
                "TX CTRL: {0} -> {1}".format(
                    command, cmd_data.hex(" ")
                )
            )
            return True
        except Exception as e:
            print(f"[Serial] 发送命令失败: {e}")
            self.log("TX CTRL 失败 {0}: {1}".format(command, e))
            return False


class SerialBridge(QThread):
    """
    串口监听线程：
    - 复用 ScreenSender 已打开的串口；或按 port/baud 自行打开；
    - 以 FF FF FF 为帧结束符；
    - 支持:
        * 0x90          → start_signal (启动识别)
        * 0x65 ...      → Nextion 触摸事件
        * ASCII 字符串  → command_signal(cmd) （例如 DEVICE_CTRL / STEPPER_FWD）
    """

    start_signal = pyqtSignal()        # 当收到启动信号时触发
    command_signal = pyqtSignal(str)   # 自定义命令（ASCII 字符串）
    log_signal = pyqtSignal(str)

    def __init__(self, port=None, baud: int = 115200, ser=None, parent=None):
        """
        - 如果 ser 已传入（外部已打开），则直接复用，不再自行打开/关闭；
        - 否则按 port/baud 自行打开。
        """
        super().__init__(parent)
        self.port = port
        self.baud = baud
        # 不用联合类型写法，兼容 3.8
        self.ser = ser  # type: Optional[serial.Serial]
        self._own = ser is None  # 我是否自己管理串口生命周期
        self._stop = False

    def open_port(self) -> bool:
        """确保串口处于打开状态。"""
        if self.ser and getattr(self.ser, "is_open", False):
            self.log_signal.emit("串口已复用（外部提供）")
            return True

        try:
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.1,
                write_timeout=0.2,
            )
            self.log_signal.emit("串口 {0} 已打开".format(self.port))
            return True
        except Exception as e:
            self.log_signal.emit("串口打开失败: {0}".format(e))
            return False

    def close_port(self):
        """如需自行管理串口，则在退出前关闭。"""
        if self._own and self.ser and self.ser.is_open:
            self.ser.close()
            self.log_signal.emit("串口 {0} 已关闭".format(self.port))

    def stop(self):
        """请求线程停止。"""
        self._stop = True

    def run(self):
        """线程主循环：按 END 切帧，解析各类命令。"""
        if not self.open_port():
            return

        buf = bytearray()

        while not self._stop:
            try:
                n = self.ser.in_waiting
                if n:
                    buf += self.ser.read(n)

                    while True:
                        i = buf.find(END)
                        if i == -1:
                            break

                        frame = bytes(buf[:i])
                        buf = buf[i + len(END):]

                        # 空帧，直接忽略
                        if not frame:
                            continue

                        # 1) 启动信号：单字节 0x90
                        if frame == b"\x90":
                            self.log_signal.emit("收到启动识别信号（0x90）")
                            self.start_signal.emit()
                            continue

                        # 2) Nextion 触摸事件：0x65 <page> <comp> <event>
                        if len(frame) >= 4 and frame[0] == 0x65:
                            p, c, ev = frame[1], frame[2], frame[3]
                            self.log_signal.emit(
                                "触摸事件 page={0} comp={1} ev={2}".format(p, c, ev)
                            )
                            # 如需通过某个触摸事件来启动识别，可以在此判断:
                            # if p == 0 and c == 3 and ev == 0:
                            #     self.start_signal.emit()
                            continue

                        # 3) 其他帧：先打印 hex，方便调试
                        self.log_signal.emit("其他帧: {0}".format(frame.hex(" ")))

                        # 3.1 尝试把“全是可打印 ASCII 的帧”当做自定义命令
                        try:
                            if all(32 <= b <= 126 for b in frame):
                                cmd = frame.decode("ascii").strip()
                                if cmd:
                                    self.log_signal.emit("自定义命令: {0}".format(cmd))
                                    self.command_signal.emit(cmd)
                        except Exception as e:
                            self.log_signal.emit("自定义命令解析失败: {0}".format(e))

                else:
                    self.msleep(10)

            except Exception as e:
                self.log_signal.emit("监听异常: {0}".format(e))
                self.msleep(50)


        # 线程退出前，按需关闭串口
        self.close_port()

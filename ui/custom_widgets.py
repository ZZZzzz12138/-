'''
集中放自定义控件
'''
import numpy as np
import math
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFrame, QSplitter,
                             QGridLayout, QGroupBox, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QTimer, Qt, QSize, QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, \
    QParallelAnimationGroup, pyqtProperty
from PyQt5.QtGui import (QImage, QPixmap, QFont, QIcon, QColor,
                         QLinearGradient, QPainter, QBrush, QPalette, QRadialGradient, QPen, QPolygonF)



class NeuralNetworkWidget(QWidget):
    """神经网络实时动画组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 200)
        self.nodes = []
        self.connections = []
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 每 50ms 更新
        self.pulse_offset = 0  # 脉冲偏移量
        self.setup_network()

    def setup_network(self):
        """设置网络节点和连接"""
        # 输入层
        for i in range(4):
            self.nodes.append({'x': 30, 'y': 30 + i * 35, 'layer': 0, 'activation': 0.3})

        # 隐藏层
        for i in range(6):
            self.nodes.append({'x': 120, 'y': 15 + i * 25, 'layer': 1, 'activation': 0.5})

        # 第二隐藏层
        for i in range(4):
            self.nodes.append({'x': 210, 'y': 30 + i * 35, 'layer': 2, 'activation': 0.7})

        # 输出层
        for i in range(3):
            self.nodes.append({'x': 270, 'y': 45 + i * 35, 'layer': 3, 'activation': 0.8})

        # 创建连接
        layer_starts = [0, 4, 10, 14]
        layer_sizes = [4, 6, 4, 3]

        for layer in range(3):
            start_idx = layer_starts[layer]
            next_start = layer_starts[layer + 1]
            for i in range(layer_sizes[layer]):
                for j in range(layer_sizes[layer + 1]):
                    self.connections.append({
                        'from': start_idx + i,
                        'to': next_start + j,
                        'weight': np.random.random() * 0.8 + 0.2
                    })

    def update_animation(self):
        """更新动画状态"""
        self.pulse_offset += 0.1

        # 更新节点激活状态
        for i, node in enumerate(self.nodes):
            base_activation = 0.3 + 0.4 * math.sin(self.pulse_offset + i * 0.3)
            node['activation'] = max(0.1, base_activation)

        self.update()

    def paintEvent(self, event):
        """绘制网络、连接和激活效果"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制连接线（随激活变换透明度和颜色）
        for conn in self.connections:
            from_node = self.nodes[conn['from']]
            to_node = self.nodes[conn['to']]
            weight = conn['weight']

            # 根据激活和权重设置颜色强度
            alpha = int(weight * from_node['activation'] * 150 + 50)
            color = QColor(0, 240, 255, alpha)
            pen = QPen(color, 1)
            painter.setPen(pen)
            painter.drawLine(from_node['x'], from_node['y'], to_node['x'], to_node['y'])

        # 绘制节点
        for node in self.nodes:
            radius = 4 + node['activation'] * 3
            intensity = int(node['activation'] * 255)

            # 发光效果
            glow_color = QColor(0, 240, 255, 80)
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(node['x'] - radius - 2, node['y'] - radius - 2,
                                (radius + 2) * 2, (radius + 2) * 2)

            # 节点核心
            core_color = QColor(255, 255, 255, intensity)
            painter.setBrush(QBrush(core_color))
            painter.drawEllipse(node['x'] - radius, node['y'] - radius,
                                radius * 2, radius * 2)


class AnimatedButton(QPushButton):
    """带动画效果的按钮"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.original_geometry = None
        self.is_hovering = False

        # 光晕效果
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(20)
        self.glow_effect.setOffset(0, 0)
        self.glow_effect.setColor(QColor(0, 240, 255, 100))
        self.setGraphicsEffect(self.glow_effect)

    def enterEvent(self, event):
        """鼠标进入时触发动画"""
        if not self.is_hovering:
            self.is_hovering = True
            if self.original_geometry is None:
                self.original_geometry = self.geometry()

            expanded = self.original_geometry.adjusted(-3, -3, 3, 3)
            self.hover_animation.setStartValue(self.geometry())
            self.hover_animation.setEndValue(expanded)
            self.hover_animation.start()

            # 增强光晕
            self.glow_effect.setBlurRadius(30)
            self.glow_effect.setColor(QColor(0, 240, 255, 150))

        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开时恢复动画"""
        if self.is_hovering:
            self.is_hovering = False
            if self.original_geometry:
                self.hover_animation.setStartValue(self.geometry())
                self.hover_animation.setEndValue(self.original_geometry)
                self.hover_animation.start()

            # 恢复光晕
            self.glow_effect.setBlurRadius(20)
            self.glow_effect.setColor(QColor(0, 240, 255, 100))

        super().leaveEvent(event)


class PulsingLabel(QLabel):
    """脉冲动画标签"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation.setDuration(1500)
        self.opacity_animation.setStartValue(0.6)
        self.opacity_animation.setEndValue(1.0)
        self.opacity_animation.setEasingCurve(QEasingCurve.InOutSine)

        # 创建循环动画
        self.animation_group = QSequentialAnimationGroup()
        self.animation_group.addAnimation(self.opacity_animation)

        reverse_animation = QPropertyAnimation(self, b"windowOpacity")
        reverse_animation.setDuration(1500)
        reverse_animation.setStartValue(1.0)
        reverse_animation.setEndValue(0.6)
        reverse_animation.setEasingCurve(QEasingCurve.InOutSine)
        self.animation_group.addAnimation(reverse_animation)

        self.animation_group.setLoopCount(-1)  # 无限循环

    def start_pulsing(self):
        """开始脉冲动画"""
        self.animation_group.start()

    def stop_pulsing(self):
        """停止脉冲动画"""
        self.animation_group.stop()
        self.setWindowOpacity(1.0)


class GradientFrame(QFrame):
    """带有渐变背景的框架"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.animation_offset = 0

        # 定时器定期更新背景效果
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_gradient)
        self.animation_timer.start(100)

    def update_gradient(self):
        """更新背景动画参数"""
        self.animation_offset += 0.02
        self.update()

    def paintEvent(self, event):
        """绘制渐变背景和网格线"""
        painter = QPainter(self)

        # 动态径向渐变
        gradient = QRadialGradient(
            self.width() * (0.5 + 0.2 * math.sin(self.animation_offset)),
            self.height() * (0.5 + 0.1 * math.cos(self.animation_offset * 1.3)),
            min(self.width(), self.height()) * 0.8
        )

        color1_intensity = int(30 + 10 * math.sin(self.animation_offset))
        color2_intensity = int(15 + 5 * math.cos(self.animation_offset * 0.7))

        gradient.setColorAt(0, QColor(color1_intensity, color1_intensity, color1_intensity + 20))  # 中心颜色
        gradient.setColorAt(0.7, QColor(10, 10, 25))  # 中间颜色
        gradient.setColorAt(1, QColor(color2_intensity, color2_intensity, color2_intensity + 15))  # 边缘颜色

        painter.fillRect(self.rect(), QBrush(gradient))

        # 绘制网格
        pen = QPen(QColor(0, 240, 255, 30))
        pen.setWidth(1)
        painter.setPen(pen)

        grid_size = 40
        for i in range(0, self.width(), grid_size):
            painter.drawLine(i, 0, i, self.height())

        for i in range(0, self.height(), grid_size):
            painter.drawLine(0, i, self.width(), i)

        super().paintEvent(event)


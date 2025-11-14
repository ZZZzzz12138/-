# ui_main_window.py
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QGridLayout,
    QGroupBox,
    QTextEdit,
    QTabWidget,
)
from PyQt5.QtCore import Qt

# 优先包内相对导入；如果直接跑该文件，则退回到同目录导入
try:
    from .custom_widgets import (
        NeuralNetworkWidget,
        AnimatedButton,
        PulsingLabel,
        GradientFrame,
    )
except ImportError:
    from custom_widgets import (
        NeuralNetworkWidget,
        AnimatedButton,
        PulsingLabel,
        GradientFrame,
    )


class MainWindowUI:
    """只负责创建控件和布局的 Mixin"""

    def setup_ui(self):
        """供 MainWindow 调用的统一入口"""
        self.create_widgets()
        self.create_control_panel_widgets()
        self.setup_layout()

    def create_widgets(self):
        """创建视觉识别主界面的组件"""
        self.title_label = PulsingLabel("智能物料视觉识别系统")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFixedHeight(70)
        self.title_label.start_pulsing()

        # 摄像头显示区域
        self.camera_frame = GradientFrame()
        self.camera_frame.setObjectName("camera_frame")
        camera_layout = QVBoxLayout(self.camera_frame)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("🔮 AI视觉系统初始化中...")
        self.camera_label.setStyleSheet(
            "font-size: 22px; color: #80E0FF; font-weight: 300;"
        )
        camera_layout.addWidget(self.camera_label)

        # 识别结果显示区域
        self.result_label = QLabel("🚀 神经网络就绪，等待启动识别...")
        self.result_label.setObjectName("result_label")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(100)

        # 处理流程组
        self.process_group = QGroupBox("AI 视觉处理管线")
        process_layout = QGridLayout(self.process_group)

        network_container = QWidget()
        network_layout = QVBoxLayout(network_container)
        self.neural_network = NeuralNetworkWidget()
        network_title = QLabel("神经网络实时状态")
        network_title.setAlignment(Qt.AlignCenter)
        network_title.setStyleSheet(
            "font-size: 14px; color: #80E0FF; margin-bottom: 10px;"
        )
        network_layout.addWidget(network_title)
        network_layout.addWidget(self.neural_network)
        network_layout.setContentsMargins(10, 10, 10, 10)

        self.process_label1 = QLabel("颜色特征提取")
        self.process_label1.setObjectName("process_label")
        self.process_label1.setAlignment(Qt.AlignCenter)

        self.process_label2 = QLabel("模式识别分类")
        self.process_label2.setObjectName("process_label")
        self.process_label2.setAlignment(Qt.AlignCenter)

        self.process_info1 = QLabel("Color Detection")
        self.process_info1.setAlignment(Qt.AlignCenter)
        self.process_info1.setStyleSheet(
            "font-size: 14px; color: #80E0FF; margin-top: 8px; font-weight: 300;"
        )

        self.process_info2 = QLabel("Pattern Classification")
        self.process_info2.setAlignment(Qt.AlignCenter)
        self.process_info2.setStyleSheet(
            "font-size: 14px; color: #80E0FF; margin-top: 8px; font-weight: 300;"
        )

        # 固定尺寸与布局
        self.process_label1.setFixedSize(280, 200)
        self.process_label2.setFixedSize(280, 200)
        self.process_info1.setFixedHeight(30)
        self.process_info2.setFixedHeight(30)

        process_layout.addWidget(network_container, 0, 0, 2, 1)
        process_layout.addWidget(self.process_label1, 0, 1)
        process_layout.addWidget(self.process_info1, 1, 1)
        process_layout.addWidget(self.process_label2, 0, 2)
        process_layout.addWidget(self.process_info2, 1, 2)
        process_layout.setColumnStretch(0, 1)
        process_layout.setColumnStretch(1, 1)
        process_layout.setColumnStretch(2, 1)
        process_layout.setSpacing(20)

        # 控制按钮
        self.start_button = AnimatedButton("启动识别")
        self.start_button.setObjectName("start_button")
        self.start_button.setMinimumHeight(55)

        self.stop_button = AnimatedButton("停止识别")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setMinimumHeight(55)
        self.stop_button.setEnabled(True)

        self.quit_button = AnimatedButton("退出系统")
        self.quit_button.setMinimumHeight(55)

        # 状态栏
        self.status_label = QLabel("状态: AI模型加载完成 | 神经网络就绪")
        self.status_label.setStyleSheet(
            """
            font-size: 14px; 
            color: #A0C0FF; 
            padding: 10px;
            background: rgba(0, 60, 120, 0.2);
            border-radius: 8px;
            border: 1px solid rgba(0, 120, 180, 0.3);
        """
        )

        self.decor_label = QLabel("◦ ◦ ◦  NEURAL  VISION  SYSTEM  ◦ ◦ ◦")
        self.decor_label.setAlignment(Qt.AlignCenter)
        self.decor_label.setStyleSheet(
            """
            font-size: 16px; 
            letter-spacing: 4px; 
            color: rgba(0, 200, 255, 0.6);
            font-weight: 300;
            margin: 10px 0;
        """
        )
        # ================== 第三页：匀速圆周运动视觉验证组件 ==================
        # 左侧：实验视频 / 实时画面显示
        self.verify_video_frame = GradientFrame()
        self.verify_video_frame.setObjectName("verify_video_frame")
        verify_video_layout = QVBoxLayout(self.verify_video_frame)
        verify_video_layout.setContentsMargins(10, 10, 10, 10)

        self.verify_video_label = QLabel("📹 在此显示毛刷旋转的实时画面 / 实验视频")
        self.verify_video_label.setAlignment(Qt.AlignCenter)
        self.verify_video_label.setStyleSheet(
            "font-size: 18px; color: #80E0FF; font-weight: 300;"
        )
        self.verify_video_label.setMinimumHeight(260)
        verify_video_layout.addWidget(self.verify_video_label)

        # 右上：视觉验证控制区
        self.verify_control_group = QGroupBox("视觉验证控制")
        verify_control_layout = QVBoxLayout(self.verify_control_group)

        self.verify_load_video_btn = AnimatedButton("载入实验视频")
        self.verify_start_capture_btn = AnimatedButton("开始采集 / 跟踪标记点")
        self.verify_analyze_btn = AnimatedButton("开始匀速性分析")
        self.verify_export_btn = AnimatedButton("导出验证报告")

        verify_control_layout.addWidget(self.verify_load_video_btn)
        verify_control_layout.addWidget(self.verify_start_capture_btn)
        verify_control_layout.addWidget(self.verify_analyze_btn)
        verify_control_layout.addWidget(self.verify_export_btn)

        self.verify_status_label = QLabel("提示：先载入视频或连接相机，再开始采集。")
        self.verify_status_label.setWordWrap(True)
        self.verify_status_label.setStyleSheet(
            "font-size: 12px; color: #A0C0FF; margin-top: 6px;"
        )
        verify_control_layout.addWidget(self.verify_status_label)

        # 右中：匀速性分析结果区
        self.verify_result_group = QGroupBox("匀速性分析结果")
        verify_result_layout = QGridLayout(self.verify_result_group)

        label_style = "font-size: 13px; color: #E8E8FF;"
        value_style = "font-size: 13px; color: #80E0FF; font-weight: 500;"

        self.avg_omega_label = QLabel("平均角速度 ω̄：")
        self.avg_omega_label.setStyleSheet(label_style)
        self.avg_omega_value = QLabel("-- rad/s")
        self.avg_omega_value.setStyleSheet(value_style)

        self.std_omega_label = QLabel("角速度标准差 σ：")
        self.std_omega_label.setStyleSheet(label_style)
        self.std_omega_value = QLabel("-- rad/s")
        self.std_omega_value.setStyleSheet(value_style)

        self.max_dev_label = QLabel("最大相对波动：")
        self.max_dev_label.setStyleSheet(label_style)
        self.max_dev_value = QLabel("-- %")
        self.max_dev_value.setStyleSheet(value_style)

        verify_result_layout.addWidget(self.avg_omega_label, 0, 0)
        verify_result_layout.addWidget(self.avg_omega_value, 0, 1)
        verify_result_layout.addWidget(self.std_omega_label, 1, 0)
        verify_result_layout.addWidget(self.std_omega_value, 1, 1)
        verify_result_layout.addWidget(self.max_dev_label, 2, 0)
        verify_result_layout.addWidget(self.max_dev_value, 2, 1)

        self.uniform_conclusion_label = QLabel("当前结论：尚未开始分析")
        self.uniform_conclusion_label.setWordWrap(True)
        self.uniform_conclusion_label.setStyleSheet(
            "font-size: 13px; color: #FFD580; margin-top: 8px;"
        )
        verify_result_layout.addWidget(self.uniform_conclusion_label, 3, 0, 1, 2)

        verify_result_layout.setColumnStretch(0, 1)
        verify_result_layout.setColumnStretch(1, 1)

        # 下方：视觉验证原理说明区
        self.verify_theory_group = QGroupBox("视觉验证原理说明")
        verify_theory_layout = QVBoxLayout(self.verify_theory_group)

        self.verify_theory_text = QTextEdit()
        self.verify_theory_text.setReadOnly(True)
        self.verify_theory_text.setStyleSheet(
            """
            QTextEdit {
                background: rgba(0, 20, 40, 0.3);
                border: 1px solid rgba(0, 180, 220, 0.3);
                border-radius: 8px;
                color: #E8E8FF;
                font-size: 12px;
                padding: 10px;
            }
        """
        )
        self.verify_theory_text.setPlainText(
            "视觉验证匀速圆周运动的基本思路：\n"
            "1. 在毛刷上做一个明显的标记点（例如贴一小块亮色贴纸）。\n"
            "2. 使用相机采集毛刷旋转的视频图像，保持固定的拍摄位置和帧率。\n"
            "3. 对每一帧图像，通过视觉算法找到标记点的位置，计算其与圆心连线的夹角 θ(t)。\n"
            "4. 按时间顺序对 θ(t) 求差分，得到离散角速度 ω(t) ≈ Δθ/Δt。\n"
            "5. 统计平均角速度、标准差和最大相对波动，判断角速度是否基本保持不变。\n"
            "6. 如果 ω(t) 在误差允许范围内波动很小，则可以认为 PID 控制下毛刷做近似匀速圆周运动。"
        )
        verify_theory_layout.addWidget(self.verify_theory_text)

    def create_control_panel_widgets(self):
        """创建右侧控制面板的组件"""
        # 步进电机控制组
        self.stepper_group = QGroupBox("步进电机控制")
        stepper_layout = QVBoxLayout(self.stepper_group)

        self.stepper_forward_btn = AnimatedButton("正转")
        self.stepper_backward_btn = AnimatedButton("反转")
        self.stepper_stop_btn = AnimatedButton("停止")
        self.stepper_stop_btn.setObjectName("stop_button")

        stepper_layout.addWidget(self.stepper_forward_btn)
        stepper_layout.addWidget(self.stepper_backward_btn)
        stepper_layout.addWidget(self.stepper_stop_btn)

        # 翻转机构控制组
        self.flip_group = QGroupBox("翻转机构控制")
        flip_layout = QVBoxLayout(self.flip_group)

        self.flip_forward_btn = AnimatedButton("物料翻转")
        self.flip_backward_btn = AnimatedButton("冲压")
        self.flip_stop_btn = AnimatedButton("底座翻转")
        self.flip_stop_btn.setObjectName("stop_button")

        flip_layout.addWidget(self.flip_forward_btn)
        flip_layout.addWidget(self.flip_backward_btn)
        flip_layout.addWidget(self.flip_stop_btn)

        # 串口通信日志
        self.serial_log_group = QGroupBox("串口通信日志")
        serial_layout = QVBoxLayout(self.serial_log_group)

        self.serial_log_text = QTextEdit()
        self.serial_log_text.setReadOnly(True)
        self.serial_log_text.setMaximumHeight(300)
        self.serial_log_text.setStyleSheet(
            """
            QTextEdit {
                background: rgba(0, 20, 40, 0.3);
                border: 1px solid rgba(0, 180, 220, 0.3);
                border-radius: 8px;
                color: #E8E8FF;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 10px;
            }
        """
        )

        # 清空日志按钮
        self.clear_log_btn = AnimatedButton("清空日志")
        self.clear_log_btn.setMaximumWidth(120)

        log_control_layout = QHBoxLayout()
        log_control_layout.addWidget(QLabel("通信记录:"))
        log_control_layout.addStretch()
        log_control_layout.addWidget(self.clear_log_btn)

        serial_layout.addLayout(log_control_layout)
        serial_layout.addWidget(self.serial_log_text)

    def setup_layout(self):
        """设置主布局结构"""
        # 控制按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.quit_button)
        button_layout.setSpacing(25)

        # 选项卡容器
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            """
            QTabWidget::pane {
                border: 2px solid rgba(0, 240, 255, 0.3);
                border-radius: 10px;
                background: rgba(0, 50, 80, 0.1);
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 120, 180, 0.6),
                    stop:1 rgba(0, 80, 120, 0.4));
                border: 1px solid rgba(0, 240, 255, 0.3);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 25px;
                color: #E8E8FF;
                font-size: 14px;
                font-weight: 500;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 180, 240, 0.8),
                    stop:1 rgba(0, 140, 200, 0.6));
                border-color: rgba(0, 255, 255, 0.5);
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 160, 220, 0.7),
                    stop:1 rgba(0, 120, 180, 0.5));
            }
        """
        )

        # 选项卡1：视觉识别界面
        vision_tab = QWidget()
        vision_layout = QVBoxLayout(vision_tab)
        vision_layout.addWidget(self.result_label)
        vision_layout.addSpacing(25)
        vision_layout.addWidget(self.process_group, 1)
        vision_layout.addSpacing(25)
        vision_layout.addLayout(button_layout)
        vision_layout.addSpacing(15)
        vision_layout.addWidget(self.status_label)
        vision_layout.setContentsMargins(20, 20, 20, 20)

        # 选项卡2：控制面板界面
        control_tab = QWidget()
        control_layout = QGridLayout(control_tab)
        control_layout.addWidget(self.stepper_group, 0, 0)
        control_layout.addWidget(self.flip_group, 0, 1)
        control_layout.addWidget(self.serial_log_group, 1, 0, 1, 2)
        control_layout.setRowStretch(1, 1)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(20)

        # 选项卡3：匀速圆周运动视觉验证界面
        verification_tab = QWidget()
        verification_layout = QGridLayout(verification_tab)

        # 左边：视频 / 实时画面
        verification_layout.addWidget(self.verify_video_frame, 0, 0, 2, 1)

        # 右上：控制区
        verification_layout.addWidget(self.verify_control_group, 0, 1)

        # 右中：结果区
        verification_layout.addWidget(self.verify_result_group, 1, 1)

        # 下方：原理说明
        verification_layout.addWidget(self.verify_theory_group, 2, 0, 1, 2)

        verification_layout.setRowStretch(0, 2)
        verification_layout.setRowStretch(1, 2)
        verification_layout.setRowStretch(2, 3)
        verification_layout.setColumnStretch(0, 3)
        verification_layout.setColumnStretch(1, 2)
        verification_layout.setContentsMargins(15, 15, 15, 15)
        verification_layout.setSpacing(20)

        # 添加选项卡
        self.tab_widget.addTab(vision_tab, "🎯 视觉识别")
        self.tab_widget.addTab(control_tab, "⚙️ 设备控制")
        self.tab_widget.addTab(verification_tab, "📐 匀速验证")

        # 主分割布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.camera_frame)
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([900, 500])
        splitter.setHandleWidth(3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.decor_label)
        main_layout.addWidget(splitter, 1)
        main_layout.setContentsMargins(15, 15, 15, 10)
        main_layout.setSpacing(5)

        self.setLayout(main_layout)

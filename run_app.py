# ui/run_app.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    font = QFont("SF Pro Display", 10)
    if not font.exactMatch():
        font = QFont("Segoe UI", 10)
    font.setWeight(QFont.Normal)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

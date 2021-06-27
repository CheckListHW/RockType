from PyQt5 import QtWidgets as widgets
from PyQt5.QtWidgets import QApplication, QMainWindow

import sys


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('poops')
        self.setGeometry(300, 250, 500, 500)

        self.text = widgets.QLabel(self)
        self.text.setText('       ')
        self.text.move(100, 250)
        self.text.width = 100

        btn = widgets.QPushButton(self)
        btn.move(100, 200)
        btn.setText('Я не лох?')
        btn.setFixedWidth(100)
        btn.clicked.connect(self.press_btn)

    def press_btn(self,):
        self.text.setText('спорно')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())
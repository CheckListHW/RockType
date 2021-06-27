from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QGridLayout

import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

from plot import PlotCanvas

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        ui = uic.loadUi('winland.ui', self)

        PlotCanvas(ui.rockTypeSA, n_plot=4)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())

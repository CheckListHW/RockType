import math
import os
import sys
import traceback

from PyQt5.QtCore import pyqtSignal, pyqtProperty
from PyQt5.QtWidgets import QApplication

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

from plot import PlotCanvas
from FZI import FZI

BASE_DIR = os.path.dirname(__file__)


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("Oбнаружена ошибка !:", tb)


sys.excepthook = excepthook


def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return 'Неизвестный метод'


class Window(QMainWindow):
    method_names = {
        'Winland r35': 'winland',
        'Lucia (RFN)': 'lucia',
        'FZI': 'fzi',
        'GHE': 'ghe',
        'Buckles': 'buckles',
        'Cuddy': 'cuddy'
    }

    def __init__(self):
        super(Window, self).__init__()
        uic.loadUi('rocktype.ui', self)
        self.calc_main_btn.clicked.connect(self.calc_main_plot)
        self.calc_rock_type_btn.clicked.connect(self.calc_rock_type)
        self.calc_RTWS_Btn.clicked.connect(self.calc_RTWS)
        self.update_chart.clicked.connect(self.update_grid)

    def calc_main_plot(self):
        self.method_names.get(self.comboBox_rocktype_3.currentText())
        calc_method = getattr(self, self.method_names.get(self.comboBox_rocktype_3.currentText()))
        calc_method()

    def update_grid(self, ):
        print(self.autoGridSize)
        print(self.autoGridSize.__dict__)
        if self.autoGridSize.checkState() == 0:
            n_rows = int(self.n_rows_CB.value())
            n_cols = int(self.n_cols_CB.value())
        else:
            n_rows, n_cols = self.plot.calc_auto_grid_size()
            self.n_rows_CB.setValue(n_rows)
            self.n_cols_CB.setValue(n_cols)
        self.plot.set_grid_size(nrows=n_rows, ncols=n_cols)
        self.plot.update_grid()

    def calc_RTWS(self):
        self.RTWS = 'current'
        if self.RTWS == 'current':
            sw = self.main.modify_current_sw
        elif self.RTWS == 'residual':
            sw = self.main.modify_residual_sw
        RTWS = self.plot.add_plot('RTWS')
        self.plot.draw_RTWS(RTWS, sw)

    def calc_rock_type(self):
        ax = self.plot.add_plot('Рок-типы', [0, 0], [0, 0])
        rock_type_borders, _ = self.plot.get_borders_main_plot()

        dots_rock_type = self.main.calculate_rock_type(rock_type_borders)

        self.plot.draw_rock_type(ax, dots_rock_type)

    def winland(self):
        print('winland')

    def lucia(self):
        print('lucia')

    def fzi(self):
        if hasattr(self, 'plot'):
            self.plot.reset()
        else:
            self.plot = PlotCanvas(self.rockTypeSA, n_plot=1, )

        self.FZI = FZI(BASE_DIR + '\Files\FES_svod.xlsx', BASE_DIR + '\Files\sw.xlsx', 'AJ', 'X', 'AC', 'AP', 'AQ')
        self.main = self.FZI
        self.plot.add_plot(get_key(self.method_names, 'fzi'),
                           self.FZI.FZI_chart_x, self.FZI.FZI_chart_y, 'click_add_border')

    def ghe(self):
        print('ghe')

    def buckles(self):
        print('buckles')

    def cuddy(self):
        print('cuddy')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())

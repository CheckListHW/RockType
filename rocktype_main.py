import os
import sys
import traceback

from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from plot import PlotCanvas

from fzi import fzi
from winland import winland
from lucia import lucia

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

    data_type = {
        'ГИС': 'gis',
        'Керн': 'kern',
        'Петрография': 'petro',
        'Пласт': 'layer',
        'Полигон': 'polygon',
        'Координаты скважин': 'coor'
    }

    def __init__(self):
        super(Window, self).__init__()
        uic.loadUi('rocktype.ui', self)
        self.calc_main_btn.clicked.connect(self.calc_main_plot)
        self.calc_rock_type_btn.clicked.connect(self.calc_rock_type)
        self.calc_RTWS_Btn.clicked.connect(self.calc_RTWS)
        self.update_chart.clicked.connect(self.update_grid)
        self.load_data_bt.clicked.connect(self.load_data)


    def calc_main_plot(self):
        if hasattr(self, 'plot'):
            self.plot.reset()
        else:
            self.plot = PlotCanvas(self.rockTypeSA, n_plot=1, )

        if not hasattr(self, 'petro_filename'):
            QMessageBox.critical(self, "Ошибка!", "Предварительно загрузите файл с петрогафией!", QMessageBox.Ok)
            return

        current_method_name = self.method_names.get(self.rocktype_CB.currentText())
        calc_method = getattr(self, current_method_name)
        calc_method()

    def update_grid(self, ):
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

        dots_rock_type = self.main.calc_rocktype(rock_type_borders)

        self.plot.draw_rock_type(ax, dots_rock_type)

    def winland(self):
        self.main = winland('C:/Users/kosac/PycharmProjects/winland_R35/data/rocktype_data.xlsx', 'A', 'B', 'C', 'D', 'E', 'F', 'G')
        #self.main = fzi(self.petro_filename, 'A', 'B', 'C', 'D', 'E', 'F', 'G')

        self.plot.add_plot(get_key(self.method_names, 'winland'),
                           self.main.method, self.main.probability, 'click_add_border')

        ax = self.plot.add_plot('R35 Winland histogram')

        ax.set_xscale('log')
        ax.plot(self.main.method, self.main.probability,  marker='.', linestyle='none')
        ax.hist(self.main.method, bins=len(self.main.method))

    def lucia(self):
        print('lucia')

    def fzi(self):
        #self.main = fzi('C:/Users/kosac/PycharmProjects/winland_R35/Files/rocktype_data.xlsx', 'A', 'B', 'C', 'D', 'E', 'F', 'G')
        self.main = fzi(self.petro_filename, 'A', 'B', 'C', 'D', 'E', 'F', 'G')

        self.plot.add_plot(get_key(self.method_names, 'fzi'),
                           self.main.probability, self.main.method, 'click_add_border')

    def ghe(self):
        print('ghe')

    def buckles(self):
        print('buckles')

    def cuddy(self):
        print('cuddy')

    def load_data(self):
        current_data_type = self.data_type.get(self.select_data_cb.currentText())
        calc_method = getattr(self, current_data_type)
        if calc_method() is not True:
            return

        out_item = self.data_info_list.findItems(self.select_data_cb.currentText(), QtCore.Qt.MatchContains)

        if len(out_item) > 0:
            out_item[0].setText(self.select_data_cb.currentText() + ': ' + self.petro_filename)
        else:
            self.data_info_list.addItem(self.select_data_cb.currentText() + ': ' + self.petro_filename)

    def gis(self):
        print('gis')

    def kern(self):
        print('kern')

    def petro(self):
        self.petro_filename = QFileDialog.getOpenFileName(self, "Выбирите файл петрографии", '',
                                                          "Excel files (*.xlsx *.xls);;All Files (*)")[0]
        if self.petro_filename != '':
            self.tab_rock_type.setEnabled(True)
            return True
        else:
            return False

    def layer(self):
        print('layer')

    def polygon(self):
        print('polygon')

    def coor(self):
        print('coor')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())

import os
import subprocess
from os import path
from os.path import dirname

import pandas as pd

os.environ['BASE_DIR'] = dirname(__file__)
BASE_DIR = os.environ['BASE_DIR']

import sys
from time import sleep
from traceback import format_exception
from threading import Thread
from PyQt5 import uic, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QCheckBox, QDialog

from plot import PlotCanvas
from rocktype_method import Winland, Fzi, Lucia
import ML as ml


def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return 'Неизвестный метод'


class ErrorWindow(QDialog):
    def __init__(self):
        super(ErrorWindow, self).__init__()
        uic.loadUi('ui/error.ui', self)
        self.error_text_tbrow.setText('Неизвестная ошибка')

    def exit(self, args):
        self.hide()
        if hasattr(self, 'exit_func'):
            self.exit_func()

    def open(self, text):
        self.show()
        self.error_text_tbrow.setText(text)


class ProgressBarThread(QThread):
    _signal = pyqtSignal(int)

    def __init__(self):
        super(ProgressBarThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        i = 0
        while True:
            sleep(0.03)
            i += 1
            self._signal.emit(i % 100)


class SettingsWindow(QMainWindow):
    def __init__(self):
        super(SettingsWindow, self).__init__()
        uic.loadUi('ui/settings.ui', self)
        self.testButton_2.clicked.connect(self.push_testButton_2)
        self.testButton_1.clicked.connect(self.push_testButton_1)
        self.buttonBox.clicked.connect(self.exit)
        self.test_frame.hide()

    def push_testButton_1(self):
        pass

    def push_testButton_2(self):
        pass

    def legend(self):
        return False if (self.legend_CB.checkState()) == 0 else True

    def trend_type(self):
        return self.trend_line_CB.currentText()

    def exit(self, args):
        self.hide()
        if hasattr(self, 'exit_func'):
            self.exit_func()

    def open_settings(self):
        self.show()

    def get_all_parametrs(self):
        return {
            'legend': self.legend(),
            'trend_type': self.trend_type(),
        }

    def set_exit_func(self, func):
        self.exit_func = func


class Window(QMainWindow):
    ml_default_column = ['GK', 'BK', 'RHOB', 'IK', 'DT', 'NGK']

    gis_skipped = ['MD', 'fzi', 'FZI', 'Fzi', 'winland', 'Winland', 'WINLAND',
                   'Глубина', 'глубина', 'Unnamed: 0']

    gis_selected = set()
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
        'Лабораторные исследования керна': 'petro',
        'Пласт': 'layer',
        'Полигон': 'polygon',
        'Датасет для ML': 'ml_train',
        'Координаты скважин': 'coor'
    }
    ml_train_filename = None

    def __init__(self):
        super(Window, self).__init__()
        uic.loadUi('ui/rocktype.ui', self)

        if not path.exists(BASE_DIR + '/Files'):
            os.mkdir(BASE_DIR + '/Files')

        self.error_window = ErrorWindow()
        self.settings = SettingsWindow()
        self.calc_main_btn.clicked.connect(self.calc_main_plot)
        self.calc_rock_type_btn.clicked.connect(self.calc_rock_type)
        self.calc_RTWS_btn.clicked.connect(self.calc_RTWS)
        self.update_chart.clicked.connect(self.update_grid)
        self.load_data_bt.clicked.connect(self.load_data)
        self.rocktype_CB.currentTextChanged.connect(self.rocktype_CB_changed)
        self.rocktype_CB_changed(self.rocktype_CB.currentText())
        self.test_btn.clicked.connect(self.test)
        self.settings_BTN.clicked.connect(self.open_settings)
        self.start_ml_btn.clicked.connect(self.start_ml)
        self.ml_pb.hide()

        self.settings.set_exit_func(self.update)

        self.test_btn.hide()

    def check_before_start_ml(self):
        if not hasattr(self, 'gis_filename'):
            self.error_window.open('Необходимо загрузить файл: ГИСы')
            return False

        if not os.path.isfile(self.gis_filename):
            self.error_window.open('По указному пути не удалось найти ('
                                   + self.gis_filename + ') файл: ГИСы')
            return False

        if not hasattr(self, 'petro_filename'):
            self.error_window.open('Необходимо загрузить файл: Лабораторные исследования керна')
            return False

        if not os.path.isfile(self.petro_filename):
            self.error_window.open('По указному пути не удалось найти ('
                                   + self.petro_filename + ') файл: Лабораторные исследования керна')
            return False

        return True

    def start_ml(self):
        if not self.check_before_start_ml():
            return

        ml_speed = {'fast': 256, 'average': 128, 'slow': 64}
        if hasattr(self, 'thread_progressbar'):
            if self.thread_progressbar.isRunning():
                self.thread_progressbar.terminate()

        self.ml = ml.ML(cor_list=self.gis_selected, petro_filename=self.petro_filename,
                        gis_filename=self.gis_filename, method_name=self.ml_method_cb.currentText(),
                        on_finished=self.ml_finished)

        self.ml_pb.show()
        self.thread_progressbar = ProgressBarThread()
        self.thread_progressbar._signal.connect(self.signal_pb)

        th = Thread(target=self.thread_ml)
        th.start()

    def thread_ml(self):
        self.thread_progressbar.start()
        try:
            self.ml.start(train_filename=self.ml_train_filename)
        except Exception as ex:
            self.debug_list.addItem(str(ex))
            self.ml_finished(finished=False)
        except:
            print(':(')

    def ml_finished(self, finished=True):
        self.thread_progressbar.terminate()
        self.ml_pb.hide()
        if finished:
            file_dir = self.ml.get_ML_url()
            self.path_file_lbl.setText(str(self.ml.get_ML_url()))
            if sys.platform == 'win32':
                os.startfile(os.path.realpath(file_dir))
            else:
                subprocess.call(["xdg-open", file_dir])

    def signal_pb(self, msg):
        self.ml_pb.setValue(int(msg))

    def test(self):
        self.plot.update_figure()

    def on_clicked(self):
        checkbox = self.sender()
        if checkbox.isChecked():
            self.gis_selected.add(checkbox.country)
        else:
            self.gis_selected.remove(checkbox.country)
        return

    def open_settings(self):
        self.settings.open_settings()

    def console_excepthook(self, exc_type, exc_value, exc_tb):
        tb = "".join(format_exception(exc_type, exc_value, exc_tb))
        self.debug_list.addItem(tb)
        print("Oбнаружена ошибка !:", tb)

    def rocktype_CB_changed(self, args):
        if args == 'Lucia (RFN)':
            self.calc_RTWS_btn.hide()
            self.calc_rock_type_btn.hide()
            self.calc_main_btn.setText('Рок-тип')
            return

        self.calc_RTWS_btn.show()
        self.calc_rock_type_btn.show()

    def update_grid(self):
        if self.autoGridSize.checkState() == 0:
            n_rows = int(self.n_rows_CB.value())
            n_cols = int(self.n_cols_CB.value())
        else:
            n_rows, n_cols = self.plot.calc_auto_grid_size()
            self.n_rows_CB.setValue(n_rows)
            self.n_cols_CB.setValue(n_cols)
        self.plot.set_grid_size(nrows=n_rows, ncols=n_cols)
        self.plot.update_grid()

    def calc_main_plot(self):
        if hasattr(self, 'plot'):
            self.plot.reset()
        else:
            self.plot = PlotCanvas(self.rockTypeSA, n_plot=1, )

        if not hasattr(self, 'petro_filename'):
            QMessageBox.critical(self, "Ошибка!", "Предварительно загрузите файл с лабораторными исследованиями керна!",
                                 QMessageBox.Ok)
            return

        current_method_name = self.method_names.get(self.rocktype_CB.currentText())
        calc_method = getattr(self, current_method_name)
        calc_method()

    def update(self):
        parametrs = self.settings.get_all_parametrs()
        for item in parametrs.items():
            self.plot.set_parametrs(item[0], item[1])

    def calc_RTWS(self):
        self.calc_rock_type()
        self.RTWS = self.settings.watersaturated_CB.currentText()
        if self.RTWS == 'current':
            sw = self.main.modify_current_sw
        elif self.RTWS == 'residual':
            sw = self.main.modify_residual_sw
        RTWS = self.plot.add_plot('RTWS')
        self.plot.draw_RTWS(RTWS, sw)

    def calc_rock_type(self):
        ax = self.plot.add_plot('Рок-типы', [0, 0], [0, 0], label_y='Проницаемость', label_x='Пористость')
        rock_type_borders, _ = self.plot.get_borders_main_plot()

        dots_rock_type = self.main.calc_rocktype(rock_type_borders)

        self.plot.draw_rock_type(ax, dots_rock_type)

    def winland(self):
        self.main = Winland(self.petro_filename, 'A', 'B', 'C', 'D', 'E', 'F', 'G')

        self.plot.add_plot(get_key(self.method_names, 'winland'),
                           self.main.method, self.main.probability, 'click_add_border')

        ax = self.plot.add_plot('R35 Winland histogram')

        ax.set_xscale('log')
        ax.plot(self.main.method, self.main.probability, marker='.', linestyle='none')
        ax.hist(self.main.method, bins=len(self.main.method))

    def lucia(self):
        self.main = Lucia(self.petro_filename, 'A', 'B', 'C', 'D', 'E', 'F', 'G')
        dots = self.main.calc_rocktype()

        ax = self.plot.add_plot(get_key(self.method_names, 'lucia'))
        self.plot.draw_rock_type_lucia(ax, dots)

    def fzi(self):
        self.main = Fzi(self.petro_filename, 'A', 'B', 'C', 'D', 'E', 'F', 'G')

        self.plot.add_plot(get_key(self.method_names, 'fzi'),
                           self.main.probability, self.main.method, 'click_add_border')

    def ghe(self):
        pass

    def buckles(self):
        pass

    def cuddy(self):
        pass

    def load_data(self):
        current_data_type = self.data_type.get(self.select_data_cb.currentText())
        calc_method = getattr(self, current_data_type)
        filename = calc_method()
        if filename is False:
            return

        out_item = self.data_info_list.findItems(self.select_data_cb.currentText(), QtCore.Qt.MatchContains)

        if len(out_item) > 0:
            out_item[0].setText(self.select_data_cb.currentText() + ': ' + filename)
        else:
            self.data_info_list.addItem(self.select_data_cb.currentText() + ': ' + filename)

    def gis(self):
        self.gis_filename = QFileDialog.getOpenFileName(self, "Выбирите файл ГИС", '',
                                                        "Excel files (*.xlsx *.xls);;All Files (*)")[0]
        if self.gis_filename != '' and self.ml_train_filename is None:
            self.draw_scroll_area_with_curves(self.gis_filename)
            return self.gis_filename
        else:
            return False

    def draw_scroll_area_with_curves(self, filename):
        for i in reversed(range(self.gridLayout_choose_gis.count())):
            self.gridLayout_choose_gis.itemAt(i).widget().deleteLater()
        gis_frame = pd.read_excel(filename)
        columns_gis_frame = set(gis_frame.columns).difference(set(self.gis_skipped))

        for name in self.ml_default_column:
            if name in columns_gis_frame:
                radiobutton = QCheckBox(name)
                radiobutton.setChecked(True)
                self.gis_selected.add(name)
                radiobutton.country = name
                radiobutton.toggled.connect(self.on_clicked)
                self.gridLayout_choose_gis.addWidget(radiobutton, 0, self.gridLayout_choose_gis.columnCount() + 1)

        for name in columns_gis_frame:
            if name not in self.ml_default_column:
                radiobutton = QCheckBox(name)
                radiobutton.setChecked(False)
                radiobutton.country = name
                radiobutton.toggled.connect(self.on_clicked)
                self.gridLayout_choose_gis.addWidget(radiobutton, 0, self.gridLayout_choose_gis.columnCount() + 1)

        self.scrollAreaWidgetContents_2.repaint()

    def kern(self):
        pass

    def petro(self):
        self.petro_filename = QFileDialog.getOpenFileName(self, "Выбирите файл лабораторных исследований керна", '',
                                                          "Excel files (*.xlsx *.xls);;All Files (*)")[0]
        if self.petro_filename != '':
            return self.petro_filename
        else:
            return False

    def ml_train(self):
        self.ml_train_filename = QFileDialog.getOpenFileName(self, "Выбирите файл который будет использовать ml", '',
                                                             "Excel files (*.xlsx *.xls);;All Files (*)")[0]
        if self.ml_train_filename != '':
            self.draw_scroll_area_with_curves(self.ml_train_filename)
            return self.ml_train_filename
        else:
            self.ml_train_filename = None
            return False

    def layer(self):
        pass

    def polygon(self):
        pass

    def coor(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.excepthook = window.console_excepthook
    window.show()
    sys.exit(app.exec_())

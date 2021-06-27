import random
from math import ceil
from PyQt5 import QtWidgets

import tkinter as tk

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QSizePolicy
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, n_plot=1, grid_size=None):
        if n_plot is None and grid_size is None:
            return
        if grid_size is None:
            self.nrows, self.ncols = self.get_grid_size(n_plot)
        else:
            self.nrows, self.ncols = grid_size

        self.fig = Figure(tight_layout=True)
        FigureCanvas.__init__(self, self.fig)

        self.canvas = FigureCanvas(self.fig)

        toolbar = NavigationToolbar(self.canvas, parent)

        #btn = QtWidgets.QPushButton(toolbar)
        #btn.setText('ччччэ')

        QIcon(":/images/open.png")

        menu = QtWidgets.QAction(toolbar)
        menu.setText('sss')
        menu.setCheckable(True)
        menu.QIcon = QIcon("sss.png")
        menu.triggered.connect(self.add_plot)
        print(menu.__dict__.get('QICon'))
        x = QtWidgets.QAction(menu)


        toolbar.addAction(x)
        mainLayout = QtWidgets.QGridLayout(parent)
        mainLayout.addWidget(toolbar)
        mainLayout.addWidget(self.canvas)

        self.plot()

    def get_grid_size(self, n_plot):
        n = ceil(n_plot ** (0.5))
        return n, n

    def add_plot(self):
        print('xxxxxxx')

    def plot(self):
        for i in range(self.nrows*self.ncols):
            data = [random.random() for i in range(25)]
            ax1 = self.figure.add_subplot(self.nrows, self.ncols, i+1)
            ax1.plot(data)
            ax1.set_title('PyQt Matplotlib Example '+str(i+1))

        self.draw()


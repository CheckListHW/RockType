import math
import random
from math import ceil

import numpy
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit

from PyQt5 import QtWidgets
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, n_plot=1, nrows=None, ncols=None, **kwargs):
        self.plot_type = {
            'click_add_border': ' (дважды ЛКМ добавить границу) (дважды ПКМ удалить)',
        }
        self.plot_click_add_border = []
        self.set_parameters(kwargs)
        self.set_grid_size(n_plot, nrows, ncols)

        FigureCanvas.__init__(self, Figure(tight_layout=True))

        self.mpl_connect('button_press_event', self.mouse_click)

        self.toolbar = NavigationToolbar(self, parent)

        self.mainLayout = QtWidgets.QGridLayout(parent)
        self.mainLayout.addWidget(self.toolbar)
        self.mainLayout.addWidget(self)

        self.plot_number()

    def reset(self):
        self.nrows = 1
        self.ncols = 1
        self.figure.clear()
        self.draw()

    def plot_number(self):
        return len(self.figure.axes) + 1

    def add_plot(self, name, x=None, y=None, plot_type=None, color=None):
        for axes in self.figure.axes:
            if axes.get_title().__contains__(name):
                return axes

        if len(self.figure.axes) + 1 > self.nrows * self.ncols:
            self.set_grid_size(len(self.figure.axes) + 1)

        name += '' if plot_type is None else self.plot_type.get(plot_type)
        ax = self.figure.add_subplot(self.nrows, self.ncols, self.plot_number(), facecolor=color)

        if x is not None and y is not None:
            ax.plot(x, y)

        ax.set_title(name)
        self.add_function_to_plot(ax, plot_type)

        if len(self.figure.axes) > self.nrows * self.ncols:
            self.set_grid_size(len(self.figure.axes))

        self.update_grid()
        return ax

    def update_grid(self):
        gs = gridspec.GridSpec(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                axes_number = row * self.ncols + col
                if axes_number < len(self.figure.axes):
                    self.figure.axes[axes_number].set_position(gs[row, col].get_position(self.figure))
                    self.figure.axes[axes_number].set_subplotspec(gs[row, col])

        self.draw()

    def add_function_to_plot(self, plot, plot_type):
        if plot_type == 'click_add_border':
            self.plot_click_add_border.append(plot)

    def edit_plot(self, type_change='add', plot=None, x=None, y=None, color=None):
        if type_change == 'add_line':
            plot.plot([x, x], [plot.dataLim.y0, plot.dataLim.y1], color=color)
        elif type_change == 'add':
            plot.plot(x, y, color=color)

        elif type_change == 'pop_last':
            if len(plot.lines) > 1:
                plot.lines.pop(-1)
        self.draw()

    def draw_rock_type(self, plot, dots):
        plot.lines = []
        rock_type_borders, rock_type_colors = self.get_borders_main_plot()
        for i in range(len(dots)):
            if i < len(rock_type_borders):
                rock_type_value = str(round(rock_type_borders[i], 2))
            else:
                rock_type_value = str(math.inf)
            plot.plot(0, 0, 'o', markersize=5, color=rock_type_colors[i],
                      label='Rock Type ' + str(i + 1) + ': ' + rock_type_value)
            for dot in dots[i]:
                plot.plot(dot[0], dot[1], 'o', markersize=4, color=rock_type_colors[i])

            d = pd.DataFrame(dots[i], columns=['x1', 'y1'])
            x1 = numpy.array(d['x1'])
            y1 = numpy.array(d['y1'])

            #линейная апросисимация
            #z = numpy.polyfit(x1, y1, 2)
            #p = numpy.poly1d(z)

            z = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t),  x1,  y1)
            xx = []
            yy = []

            for i1 in range(len(x1)):
                xx.append(x1[i1])
                yy.append(self.f(z[0][0], z[0][1], x1[i1]))

            rSquare = self.rSquare(yy, y1)
            plot.plot(sorted(xx), sorted(yy), "-", color=rock_type_colors[i],
                      label="y=%.4f*e^(%.3f*x)|R^2=%.3f" % (z[0][0], z[0][1], rSquare))

        plot.legend(loc="lower right")
        plot.set_yscale('log')
        self.draw()

    def f(self, a, b, x):
        return a * math.exp(b * x)

    def rSquare(self, estimations, measureds):
        """ Compute the coefficient of determination of random data.
        This metric gives the level of confidence about the model used to model data"""
        SEE = ((np.array(measureds) - np.array(estimations)) ** 2).sum()
        mMean = np.array(measureds).sum() / float(len(measureds))
        dErr = ((np.array(measureds) - mMean) ** 2).sum()

        print(SEE)
        print(dErr)


        '''
        print(SEE)
        mMean = (np.array(measureds)).sum() / float(len(measureds))
        print(mMean)
        dErr = ((mMean - measureds)).sum()
        print(dErr)
        '''
        return 1 - (SEE / dErr)

    def draw_RTWS(self, plot, dots_x):
        plot.lines = []
        borders, colors = self.get_borders_main_plot()

        for rock_type_n in range(1, len(borders) + 1):
            if rock_type_n - 1 < len(borders):
                rock_type_value = str(round(borders[rock_type_n - 1], 2))
            else:
                rock_type_value = str(math.inf)

            # plot.plot(1, 1, 'o', markersize=0, color=colors[rock_type_n - 1],
            #          label='Rock Type ' + str(rock_type_n) + ': ' + rock_type_value)

            step = len(dots_x) / (len(borders) + 1)
            plot.plot([step * rock_type_n, step * (rock_type_n + 1)], [rock_type_n, rock_type_n],
                      '-', markersize=1, color=colors[rock_type_n - 1],
                      label='Rock Type ' + str(rock_type_n) + ': ' + rock_type_value)
            plot.legend(loc="lower right")

        self.draw()

    def get_borders_main_plot(self):
        dots = []
        color = []
        for axes in self.figure.axes:
            if axes in self.plot_click_add_border:
                for line in axes.lines:
                    if len(line._x) < 3:
                        dots.append(line._x[0])
                        color.append(line.get_color())
                dots.append(math.inf)
                color.append(axes.lines[0].get_color())
                return dots, color

    def plot(self):
        for i in range(self.nrows * self.ncols):
            data = [random.random() for i in range(25)]
            ax = self.figure.add_subplot(self.nrows, self.ncols, i + 1)
            ax.plot(data)
            ax.set_title('PyQt Matplotlib Example ' + str(i + 1))
        self.draw()

    def set_parameters(self, kwargs):
        for key, value in kwargs.items():
            if key in self.parametrs:
                self.parametrs[key] = value

    def set_grid_size(self, n_plot=None, nrows=None, ncols=None):
        if n_plot is not None:
            self.nrows, self.ncols = self.calc_grid_size(n_plot)
        if nrows is not None and ncols is not None:
            self.nrows = nrows
            self.ncols = ncols

    def mouse_click(self, event):
        if event.inaxes in self.plot_click_add_border and event.dblclick is True:
            if event.button == 3:
                event.inaxes.set_title(event.inaxes.get_title().replace('(дважды ПКМ удалить)', ''))
                self.edit_plot('pop_last', event.inaxes)
            elif event.button == 1:
                x, y = event.xdata, event.ydata
                event.inaxes.set_title(event.inaxes.get_title().replace('(дважды ЛКМ добавить границу)', ''))
                self.edit_plot('add_line', event.inaxes, x, y)

    def calc_auto_grid_size(self):
        return self.calc_grid_size(len(self.figure.axes))

    def calc_grid_size(self, n_plot):
        n = ceil(n_plot ** (0.5))
        if (n - 1) * n >= n_plot:
            return n - 1, n
        return n, n

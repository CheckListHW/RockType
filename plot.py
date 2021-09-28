from math import ceil, inf, exp
from random import random

from numpy import array, exp, polyfit, poly1d
from pandas import DataFrame
from scipy.optimize import curve_fit

from PyQt5 import QtWidgets
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import data_poro


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, n_plot=1, nrows=None, ncols=None, **kwargs):
        self.parametrs = {'trend_type': 'exp'}

        for key, value in kwargs.items():
            if key in self.parametrs:
                self.parametrs[key] = value

        self.plot_type = {
            'click_add_border': ' (дважды ЛКМ добавить границу) (дважды ПКМ удалить)',
        }
        self.plot_click_add_border = []
        self.set_parameters(kwargs)
        self.set_grid_size(n_plot, nrows, ncols)

        FigureCanvasQTAgg.__init__(self, Figure(tight_layout=True))

        self.mpl_connect('button_press_event', self.mouse_click)

        self.toolbar = NavigationToolbar2QT(self, parent)

        self.mainLayout = QtWidgets.QGridLayout(parent)
        self.mainLayout.addWidget(self.toolbar)
        self.mainLayout.addWidget(self)

        self.plot_number()

    def get_parametrs(self, name):
        if self.parametrs.get(name) is not None:
            return self.parametrs.get(name)

    def set_parametrs(self, name, value):
        self.parametrs[name] = value

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

    def update_figure(self):
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
                rock_type_value = str(inf)
            plot.plot(0, 0, 'o', markersize=5, color=rock_type_colors[i],
                      label='Rock Type ' + str(i + 1) + ': ' + rock_type_value)
            for dot in dots[i]:
                plot.plot(dot[0], dot[1], 'o', markersize=4, color=rock_type_colors[i])

            d = DataFrame(dots[i], columns=['x1', 'y1'])
            x1 = array(d['x1'])
            y1 = array(d['y1'])

            if self.get_parametrs('trend_type') == 'exp':
                z = curve_fit(lambda t, a, b: a * exp(b * t), x1, y1)
                xx = []
                yy = []

                for i1 in range(len(x1)):
                    xx.append(x1[i1])
                    yy.append(self.f(z[0][0], z[0][1], x1[i1]))

                rSquare = self.rSquare(yy, y1)
                plot.plot(sorted(xx), sorted(yy), "-", color=rock_type_colors[i],
                          label="y=%.4f*e^(%.3fx)|R^2=%.3f" % (z[0][0], z[0][1], rSquare))

            if self.get_parametrs('trend_type') == 'line':
                z = polyfit(x1, y1, 1)
                p = poly1d(z)
                xx = []
                yy = []
                for i1 in range(len(x1)):
                    if p(x1[i1]) > min(y1):
                        xx.append(x1[i1])
                        yy.append(p(x1[i1]))
                plot.plot(sorted(xx), sorted(yy), "-", color=rock_type_colors[i],
                          label="y=%.6fx+(%.6f)" % (z[0], z[1]))

        self.show_legend(plot)

        plot.set_xscale('log')
        plot.set_yscale('log')
        self.draw()

    def show_legend(self, plot):
        if self.get_parametrs('legend') is True:
            plot.legend(loc="lower right")
        else:
            plot.legend().set_visible(False)

    def draw_rock_type_lucia(self, plot, dots):
        plot.lines = []
        plot.plot(data_poro.poro_05, data_poro.perm_05, ':', color='black')
        plot.plot(data_poro.poro_15, data_poro.perm_15, ':', color='black')
        plot.plot(data_poro.poro_25, data_poro.perm_25, ':', color='black')
        plot.plot(data_poro.poro_4, data_poro.perm_4, ':', color='black')
        for key in dots.keys():
            plot.plot(dots[key][0], dots[key][1], 'o', markersize=4, color=key)
            if len(dots[key][0]) > 1 and len(dots[key][1]) > 1:
                x1 = array(dots[key][0])
                y1 = array(dots[key][1])

                if self.get_parametrs('trend_type') == 'exp':
                    z = curve_fit(lambda t, a, b: a * exp(b * t), x1, y1)
                    xx = []
                    yy = []

                    for i1 in range(len(x1)):
                        xx.append(x1[i1])
                        yy.append(self.f(z[0][0], z[0][1], x1[i1]))

                    rSquare = self.rSquare(yy, y1)
                    plot.plot(sorted(xx), sorted(yy), "-", color=key,
                              label="y=%.4f*e^(%.3fx)|R^2=%.3f" % (z[0][0], z[0][1], rSquare))

                elif self.get_parametrs('trend_type') == 'line':
                    z = polyfit(x1, y1, 1)
                    p = poly1d(z)
                    xx = []
                    yy = []
                    for i1 in range(len(x1)):
                        if p(x1[i1]) > min(y1):
                            xx.append(x1[i1])
                            yy.append(p(x1[i1]))
                    plot.plot(sorted(xx), sorted(yy), "-", color=key,
                              label="y=%.6fx+(%.6f)" % (z[0], z[1]))

        self.show_legend(plot)
        plot.set_xscale('log')
        plot.set_yscale('log')
        self.draw()

    def f(self, a, b, x):
        return a * exp(b * x)

    def rSquare(self, estimations, measureds):
        SEE = ((array(measureds) - array(estimations)) ** 2).sum()
        mMean = array(measureds).sum() / float(len(measureds))
        dErr = ((array(measureds) - mMean) ** 2).sum()
        return 1 - (SEE / dErr)

    def draw_RTWS(self, plot, dots_x):
        plot.lines = []
        borders, colors = self.get_borders_main_plot()

        for rock_type_n in range(0, len(borders)):
            if rock_type_n < len(borders):
                rock_type_value = str(round(borders[rock_type_n - 1], 2))
            else:
                rock_type_value = str(inf)
            plot.plot(dots_x[rock_type_n], [rock_type_n]*(len(dots_x[rock_type_n])),
                      '-', markersize=1, color=colors[rock_type_n],
                      label='Rock Type ' + str(rock_type_n + 1) + ': ' + rock_type_value)

            self.show_legend(plot)

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
                dots.append(inf)
                color.append(axes.lines[0].get_color())
        return dots, color

    def plot(self):
        for i in range(self.nrows * self.ncols):
            data = [random() for i in range(25)]
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

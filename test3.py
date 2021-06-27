from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure

import tkinter as tk
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()

        figure1 = Figure()
        canvas1 = FigureCanvas(figure1)
        figure1.add_subplot(111)



        #btn = QtWidgets.QPushButton(toolbar)
        #btn.move(100, 0)
        button = tk.Button(master=toolbar, text='Quit')#, command=_quit)
        button.pack(side=tk.RIGHT)

        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(toolbar,0,0)
        mainLayout.addWidget(canvas1,1,0)

        self.setLayout(mainLayout)
        self.setWindowTitle("Flow Layout")

if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWin = Window()
    mainWin.show()
    sys.exit(app.exec_())
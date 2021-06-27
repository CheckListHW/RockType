from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1874, 882)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1801, 821))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_data_load = QtWidgets.QWidget()
        self.tab_data_load.setObjectName("tab_data_load")
        self.comboBox_data = QtWidgets.QComboBox(self.tab_data_load)
        self.comboBox_data.setEnabled(True)
        self.comboBox_data.setGeometry(QtCore.QRect(10, 10, 101, 22))
        self.comboBox_data.setObjectName("comboBox_data")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.pushButton_data_enter = QtWidgets.QPushButton(self.tab_data_load)
        self.pushButton_data_enter.setGeometry(QtCore.QRect(120, 10, 121, 23))
        self.pushButton_data_enter.setObjectName("pushButton_data_enter")
        self.pushButton_data_rate = QtWidgets.QPushButton(self.tab_data_load)
        self.pushButton_data_rate.setGeometry(QtCore.QRect(250, 10, 121, 23))
        self.pushButton_data_rate.setObjectName("pushButton_data_rate")
        self.tabWidget.addTab(self.tab_data_load, "")
        self.tab_rock_type = QtWidgets.QWidget()
        self.tab_rock_type.setObjectName("tab_rock_type")
        self.scrollArea = QtWidgets.QScrollArea(self.tab_rock_type)
        self.scrollArea.setGeometry(QtCore.QRect(10, 70, 1781, 731))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1758, 729))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.frame = QtWidgets.QFrame(self.tab_rock_type)
        self.frame.setGeometry(QtCore.QRect(10, 0, 1771, 61))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.comboBox_rocktype = QtWidgets.QComboBox(self.frame)
        self.comboBox_rocktype.setGeometry(QtCore.QRect(10, 10, 101, 22))
        self.comboBox_rocktype.setObjectName("comboBox_rocktype")
        self.comboBox_rocktype.addItem("")
        self.comboBox_rocktype.addItem("")
        self.comboBox_rocktype.addItem("")
        self.comboBox_rocktype.addItem("")
        self.comboBox_rocktype.addItem("")
        self.comboBox_rocktype.addItem("")
        self.frame.raise_()
        self.scrollArea.raise_()
        self.tabWidget.addTab(self.tab_rock_type, "")
        self.tab_map = QtWidgets.QWidget()
        self.tab_map.setObjectName("tab_map")
        self.tabWidget.addTab(self.tab_map, "")
        self.tab_ml = QtWidgets.QWidget()
        self.tab_ml.setObjectName("tab_ml")
        self.tabWidget.addTab(self.tab_ml, "")
        self.tab_cluster = QtWidgets.QWidget()
        self.tab_cluster.setObjectName("tab_cluster")
        self.tabWidget.addTab(self.tab_cluster, "")
        self.tab_pad = QtWidgets.QWidget()
        self.tab_pad.setObjectName("tab_pad")
        self.tabWidget.addTab(self.tab_pad, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1874, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox_data.setItemText(0, _translate("MainWindow", "ГИС"))
        self.comboBox_data.setItemText(1, _translate("MainWindow", "Керн"))
        self.comboBox_data.setItemText(2, _translate("MainWindow", "Петрография"))
        self.comboBox_data.setItemText(3, _translate("MainWindow", "Пласт"))
        self.comboBox_data.setItemText(4, _translate("MainWindow", "Полигон"))
        self.comboBox_data.setItemText(5, _translate("MainWindow", "Координаты скважин"))
        self.pushButton_data_enter.setText(_translate("MainWindow", "Выбрать данные"))
        self.pushButton_data_rate.setText(_translate("MainWindow", "Оценить данные"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_data_load), _translate("MainWindow", "Загрузка данных"))
        self.comboBox_rocktype.setItemText(0, _translate("MainWindow", "Winland r35"))
        self.comboBox_rocktype.setItemText(1, _translate("MainWindow", "Lucia (RFN)"))
        self.comboBox_rocktype.setItemText(2, _translate("MainWindow", "FZI"))
        self.comboBox_rocktype.setItemText(3, _translate("MainWindow", "GHE"))
        self.comboBox_rocktype.setItemText(4, _translate("MainWindow", "Buckles"))
        self.comboBox_rocktype.setItemText(5, _translate("MainWindow", "Cuddy"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_rock_type), _translate("MainWindow", "Рок типирование"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_map), _translate("MainWindow", "Карта"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_ml), _translate("MainWindow", "Машинное обучение"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_cluster), _translate("MainWindow", "Класстеризация"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_pad), _translate("MainWindow", "Планшет"))

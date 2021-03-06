# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(270, 10, 580, 25))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.conditionlabel = QtWidgets.QLabel(self.centralwidget)
        self.conditionlabel.setGeometry(QtCore.QRect(20, 10, 240, 20))
        self.conditionlabel.setObjectName("conditionlabel")
        self.constant_sourceviewlabel = QtWidgets.QLabel(self.centralwidget)
        self.constant_sourceviewlabel.setGeometry(QtCore.QRect(170, 45, 100, 16))
        self.constant_sourceviewlabel.setObjectName("constant_sourceviewlabel")
        self.constant_processedviewlabel = QtWidgets.QLabel(self.centralwidget)
        self.constant_processedviewlabel.setGeometry(QtCore.QRect(610, 45, 150, 16))
        self.constant_processedviewlabel.setObjectName("constant_processedviewlabel")
        self.graphicslabel_Source = QtWidgets.QLabel(self.centralwidget)
        self.graphicslabel_Source.setGeometry(QtCore.QRect(30, 70, 350, 350))
        self.graphicslabel_Source.setText("")
        self.graphicslabel_Source.setObjectName("graphicslabel_Source")
        self.graphicslabel_Processed = QtWidgets.QLabel(self.centralwidget)
        self.graphicslabel_Processed.setGeometry(QtCore.QRect(470, 70, 350, 350))
        self.graphicslabel_Processed.setText("")
        self.graphicslabel_Processed.setObjectName("graphicslabel_Processed")
        self.graphicslabel_Original = QtWidgets.QLabel(self.centralwidget)
        self.graphicslabel_Original.setGeometry(QtCore.QRect(900, 70, 350, 350))
        self.graphicslabel_Original.setText("")
        self.graphicslabel_Original.setObjectName("graphicslabel_Original")
        self.constant_originalviewlabel = QtWidgets.QLabel(self.centralwidget)
        self.constant_originalviewlabel.setGeometry(QtCore.QRect(1030, 45, 150, 16))
        self.constant_originalviewlabel.setObjectName("constant_originalviewlabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
        self.menuPreview = QtWidgets.QMenu(self.menubar)
        self.menuPreview.setObjectName("menuPreview")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSelect_ROI = QtWidgets.QAction(MainWindow)
        self.actionSelect_ROI.setObjectName("actionSelect_ROI")
        self.actionStart = QtWidgets.QAction(MainWindow)
        self.actionStart.setObjectName("actionStart")
        self.actionAbsort = QtWidgets.QAction(MainWindow)
        self.actionAbsort.setObjectName("actionAbsort")
        self.actionPlay = QtWidgets.QAction(MainWindow)
        self.actionPlay.setObjectName("actionPlay")
        self.actionStop = QtWidgets.QAction(MainWindow)
        self.actionStop.setObjectName("actionStop")
        self.menuFile.addAction(self.actionOpen)
        self.menuAnalysis.addAction(self.actionSelect_ROI)
        self.menuAnalysis.addSeparator()
        self.menuAnalysis.addAction(self.actionStart)
        self.menuAnalysis.addAction(self.actionAbsort)
        self.menuPreview.addAction(self.actionPlay)
        self.menuPreview.addAction(self.actionStop)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuPreview.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Mircotubule Project"))
        self.conditionlabel.setText(_translate("MainWindow", "Current Condition: IDLE"))
        self.constant_sourceviewlabel.setText(_translate("MainWindow", "Source Video"))
        self.constant_processedviewlabel.setText(_translate("MainWindow", "Processed Video"))
        self.constant_originalviewlabel.setText(_translate("MainWindow", "Original Video"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))
        self.menuPreview.setTitle(_translate("MainWindow", "Preview"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSelect_ROI.setText(_translate("MainWindow", "Select ROI"))
        self.actionStart.setText(_translate("MainWindow", "Start"))
        self.actionAbsort.setText(_translate("MainWindow", "Absort"))
        self.actionPlay.setText(_translate("MainWindow", "Play"))
        self.actionStop.setText(_translate("MainWindow", "Stop"))


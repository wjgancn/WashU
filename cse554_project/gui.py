from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer, QPointF
from PyQt5.QtGui import QPen, QPainter, QColor
import PyQt5

import matplotlib.pyplot as plt
from mainwindow import Ui_MainWindow  # Auto Produced by `pyuic` command
import numpy as np
import data_loader
from scipy import misc
import cv2 as cv
import algorithm as alg


class GUI(Ui_MainWindow, QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self.setupUi(self)

        # Additional UI Step Up
        self.timer = QTimer()

        self.data = None
        self.data_index = 0

        # If Draw Press Point in graphicsView_Source
        self.drawPressPoint = False
        self.graphicslabel_Source.mousePressEvent = self.capture_mousePressEvent

        # For threadshold
        self.plot_data = []

        # Set Slot
        self.setSlot()

        self.show()

    def setSlot(self):
        # System Slot
        self.timer.timeout.connect(self.timer_Slot)

        self.actionOpen.triggered.connect(self.actionOpen_Slot)
        self.actionPlay.triggered.connect(self.actionPlay_Slot)
        self.actionStop.triggered.connect(self.actionStop_Slot)
        self.actionSelect_ROI.triggered.connect(self.actionSelect_Slot)
        self.actionStart.triggered.connect(self.actionStart_Slot)

# System Slot Function
    def capture_mousePressEvent(self, ev: QtGui.QMouseEvent):

        if (self.data is not None) and (self.drawPressPoint is True):
            # Set Pixmap in Source Label
            pixmap_cur = self.ReadImageAsQImageObjectinSource(self.ConvertToPath(self.data.read_frame_nparray(self.data_index)))

            painter_cur = QPainter()
            painter_cur.begin(pixmap_cur)

            pen_cur = QPen()
            pen_cur.setColor(QColor(255, 0, 0))  # RGB Channel, which means pure red
            pen_cur.setWidth(3)
            painter_cur.setPen(pen_cur)

            painter_cur.drawPoint(ev.x(), ev.y())
            painter_cur.end()

            self.graphicslabel_Source.setPixmap(pixmap_cur)

            # Important !!!! Exchange X and Y
            index_x = ev.y()
            index_y = ev.x()

            self.im = misc.imread(self.ConvertToPath(self.data.read_frame_nparray(self.data_index)))
            label, self.line, self.centerpoint = alg.FindLineAndLabel(self.im, index_x, index_y)

            self.showVideoinFrameInProcessed(label)
            self.drawPressPoint = False

    def timer_Slot(self):
        self.data_index += 1

        if self.data_index >= self.data.tiffarray_maxframe:
            self.data_index = 0
            self.plot_data = np.array(self.plot_data)
            plt.plot(self.plot_data)
            plt.title('Changing of the Length')
            plt.savefig('PlotResult.png')
            self.plot_data = []
            self.timer.stop()

        self.showVideoinFrameInSource(self.data.read_frame_nparray(self.data_index))
        self.showVideoinFrameInOriginal(self.data.read_frame_nparray_original(self.data_index))

        self.setProgressBar(100 * self.data_index / self.data.tiffarray_maxframe)

        self.im = misc.imread(self.ConvertToPath(self.data.read_frame_nparray(self.data_index)))
        label, self.line, self.centerpoint, length = alg.UpdateLineAndLabel(self.im, self.line, self.centerpoint)
        self.plot_data.append(length)

        self.showVideoinFrameInProcessed(label)

    # User Slot Action
    def actionOpen_Slot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*.*)", options=options)

        self.data = data_loader.TiffVideoLoader(filename)
        self.showCondition("DATA READ")

        self.showVideoinFrameInSource(self.data.read_frame_nparray(self.data_index))
        self.showVideoinFrameInOriginal(self.data.read_frame_nparray_original(self.data_index))

    def actionPlay_Slot(self):
        if self.data is None:
            self.showCondition("READ DATA FIRST")
            return 0

        self.showCondition("PLAYING")
        self.timer.start(int(1000/12))

    def actionStop_Slot(self):
        if self.data is None:
            self.showCondition("READ DATA FIRST")
            return 0

        self.showCondition("STOP")
        self.timer.stop()
        self.setProgressBar(0)
        self.data_index = 0
        self.showVideoinFrameInSource(self.data.read_frame_nparray(self.data_index))

    def actionSelect_Slot(self):
        self.showCondition("SELECT POINT")
        self.drawPressPoint = True

    def actionStart_Slot(self):
        self.data_index += 1

        if self.data_index >= self.data.tiffarray_maxframe:
            self.data_index = 0
            plt.plot(self.plot_data)
            plt.show()
            self.plot_data = []
            self.timer.stop()

        self.showVideoinFrameInSource(self.data.read_frame_nparray(self.data_index))
        self.showVideoinFrameInOriginal(self.data.read_frame_nparray_original(self.data_index))

        self.setProgressBar(100 * self.data_index / self.data.tiffarray_maxframe)

        self.im = misc.imread(self.ConvertToPath(self.data.read_frame_nparray(self.data_index)))
        label, self.line, self.centerpoint, length = alg.UpdateLineAndLabel(self.im, self.line, self.centerpoint)
        self.plot_data.append(length)

        self.showVideoinFrameInProcessed(label)

# Convenience Function
    def showCondition(self, condition_string="IDLE"):
        self.conditionlabel.setText("Current Condition: " + condition_string)

    def showVideoinFrameInSource(self, nparray):
        path = self.ConvertToPath(nparray)
        frame = self.ReadImageAsQImageObjectinSource(path)
        self.graphicslabel_Source.setPixmap(frame)

    def showVideoinFrameInProcessed(self, nparray):
        path = self.ConvertToPath(nparray)
        frame = self.ReadImageAsQImageObjectinProcessed(path)
        self.graphicslabel_Processed.setPixmap(frame)

    def showVideoinFrameInOriginal(self, nparray):
        path = self.ConvertToPath(nparray)
        frame = self.ReadImageAsQImageObjectinProcessed(path)
        self.graphicslabel_Original.setPixmap(frame)

    def setProgressBar(self, value):
        self.progressBar.setValue(int(value))

    def ReadImageAsQImageObjectinSource(self, path):
        frame = QtGui.QImage(path)
        frame = frame.scaled(self.graphicslabel_Source.width(), self.graphicslabel_Source.height(),
                             PyQt5.QtCore.Qt.IgnoreAspectRatio)
        frame = QtGui.QPixmap(frame)
        return frame

    def ReadImageAsQImageObjectinProcessed(self, path):
        frame = QtGui.QImage(path)
        frame = frame.scaled(self.graphicslabel_Processed.width(), self.graphicslabel_Processed.height(),
                             PyQt5.QtCore.Qt.IgnoreAspectRatio)
        frame = QtGui.QPixmap(frame)
        return frame

    def ConvertToPath(self, nparray):
        data = misc.imresize(nparray, (350, 350))
        misc.imsave('./temp/temp.png', data)
        return './temp/temp.png'

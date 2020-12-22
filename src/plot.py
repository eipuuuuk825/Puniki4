import numpy as np
# グラフ
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


class PlotGraph:
    def __init__(self):
        # UIを設定
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('program frequency')
        self.win.resize(500, 300)
        self.win.move(840, 400)
        self.plt = self.win.addPlot()
        self.plt.setYRange(0, 60)
        self.plt.showGrid(x=True, y=True)   # グリッドを表示
        self.plt.setLabel('left', u'動作周波数', units=u'Hz')
        self.curve = self.plt.plot(pen=(255, 255, 255))

        # データを更新する関数を呼び出す時間を設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(500)
        self.data = np.zeros(100)

    def update(self):
        self.curve.setData(self.data)

    # プロットするデータを追加
    def append(self, data):
        self.data = np.delete(self.data, 0)
        self.data = np.append(self.data, data)



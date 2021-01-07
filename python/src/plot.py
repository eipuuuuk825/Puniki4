import numpy as np
# グラフ
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


class PlotGraph:
    def __init__(self, title, label, max_y, size, pos):
        self.max_val = 0        # プロットする値の最大値
        self.auto_scale = False # 縦軸の範囲を最大値に合わせる
        if max_y == -1:
            self.auto_scale = True
            max_y = 1

        # UIを設定
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle(title)
        self.win.resize(size[0], size[1])
        self.win.move(pos[0], pos[1])
        self.plt = self.win.addPlot()
        self.plt.setYRange(0, max_y)
        self.plt.showGrid(x=True, y=True)   # グリッドを表示
        self.plt.setLabel('left', label)
        self.curve = self.plt.plot(pen=(255, 255, 255))

        # データを更新する関数を呼び出す時間を設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(500)
        self.data = np.zeros(100)

    def update(self):
        if self.auto_scale:
            self.plt.setYRange(0, self.max_val)
        self.curve.setData(self.data)

    # プロットするデータを追加
    def append(self, data):
        if self.auto_scale:
            self.max_val = max(self.max_val, data)  # 最大値を更新
        self.data = np.delete(self.data, 0)
        self.data = np.append(self.data, data)

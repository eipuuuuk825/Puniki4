# 自分の
import transform as tr
import param as pa
import judge
import plot
import capture as cap
import cv2
import numpy as np
import time
import math
import sys
import datetime  # 現在時刻取得
# windows API
from ctypes import *
user32 = windll.user32

path_data_output = "../../data_raw/stage4_2021-01-09/"


def main():
    # グラフ
    g_f = plot.PlotGraph('program frequency',
                         u'動作周波数 [Hz]',
                         max_y=60,
                         size=(500, 200),
                         pos=(840, 350))
    g_ball = plot.PlotGraph('ball y',
                            u'y [px]',
                            max_y=-1,
                            size=(500, 200),
                            pos=(840, 600))

    # 初期化処理
    capture = cap.init_camera()  # カメラ初期設定
    time_s = time.time()        # 処理時間
    time_sum = 0                # 終了処理関係
    time_throw = 0              # ボールが投げられてから経過した時間
    ball = Ball()               # ボール管理

    # フラグ
    start_flag = False          # 最初のジャッジが行われるまで開始しない
    detect_flag = 0             # ボールの検出を行うか
    judge_pre = "None"
    judge_curr = "None"

    # while(time_sum < 5):
    while(True):
        # キー入力
        hit_key = cv2.waitKey(1) & 0xFF
        if hit_key == ord('q'):
            break

        # 画像を用意
        img_cap, img_main, img_bin, img_tmp = cap.prepare_img(capture)

        # swing.process(ball)               # スイング処理
        judge_pre = judge_curr              # judge を更新
        judge_curr = judge.judge(img_main)  # 球のジャッジ

        # 最初のジャッジが行われるまで開始しない
        if judge_curr != "None":
            start_flag = True
        if start_flag == False:
            continue

        # ボール位置を取得（main 座標系）
        ball.get_pos(img_bin)

        # 終了
        if judge_curr == "Finish":
            ball.clear()
            detect_flag = 0
            start_flag = False
            continue

        # detect_flag の管理
        if judge_pre != "None" and judge_curr == "None":
            detect_flag = 1
        if detect_flag == 1 and ball.tmp_pos[1] != (0, 0):
            detect_flag = 2
        if detect_flag == 2 and ball.tmp_pos[1] == (0, 0):
            ball.output()
            ball.clear()
            detect_flag = 0

        # ボールが投げられてからの時間を計測
        if detect_flag != 2:
            time_throw = time.time()

        # ボール位置を保存
        if detect_flag == 2:
            g_ball.append(ball.tmp_pos[1][1])
            ball.append(time.time() - time_throw)

        #
        # ウィンドウ関連
        #
        # 処理時間
        time_e = time.time()
        freq = (int)(1.0 / (time_e - time_s))
        time_sum += time_e - time_s
        time_s = time_e
        g_f.append(freq)   # 描画用にデータを追加

        # ウィンドウに文字列表示
        cv2.rectangle(img_main, (5, 5), (150, 75), pa.WHITE, thickness=-1)
        cv2.putText(img_main, judge_curr, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, pa.RED, 2)
        # ボールの検出領域
        cv2.rectangle(img_main,
                      (pa.RECT_BALL_IN_MAIN[0], pa.RECT_BALL_IN_MAIN[1]),
                      (pa.RECT_BALL_IN_MAIN[0]+pa.RECT_BALL_IN_MAIN[2],
                       pa.RECT_BALL_IN_MAIN[1]+pa.RECT_BALL_IN_MAIN[3]),
                      pa.WHITE, thickness=2)
        # 検出したボールを表示
        cv2.circle(img_main, ball.tmp_pos[1], 1, pa.RED, 5)

        # ウィンドウ表示
        cv2.imshow('win', img_main)
        cv2.moveWindow('win', 10, 10)
        cv2.imshow('hsv', img_bin)
        cv2.moveWindow('hsv', 840, 10)
    #
    # 終了処理
    #
    capture.release()
    cv2.destroyAllWindows()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


#
# ボール
#
class Ball:
    def __init__(self):
        self.data = []       # (x, y, t) のリスト
        # 求めた位置を一時的に保管（0 : previous, 1 : current）
        self.tmp_pos = [(0, 0), (0, 0)]

    def clear(self):
        self.data.clear()

    def append(self, time):
        self.data.append((self.tmp_pos[1][0], self.tmp_pos[1][1], time))

    def len(self):
        return len(self.data)

    #
    # ボール位置を求める（main 座標系で取得）
    #
    def get_pos(self, img_bin):
        self.tmp_pos[0] = self.tmp_pos[1]   # previous を保存
        self.tmp_pos[1] = (0, 0)            # current を初期化
        mu = cv2.moments(img_bin, False)
        # 重心を計算
        if mu["m00"] != 0:
            self.tmp_pos[1] = (int(mu["m10"]/mu["m00"]),
                               int(mu["m01"]/mu["m00"]))
            self.tmp_pos[1] = tr.ball_main(self.tmp_pos[1])  # ball -> main へ変換

    #
    # データを出力
    #
    def output(self):
        filename = str(datetime.datetime.now()).replace(":", ".")
        print(filename)
        file = path_data_output + filename + ".csv"
        with open(file, "w") as fileobj:
            for i in self.data:
                for j in i:
                    fileobj.write(str(j)+",")
                fileobj.write("\n")


if __name__ == '__main__':
    main()

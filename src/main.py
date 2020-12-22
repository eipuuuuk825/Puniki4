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
# windows API
from ctypes import *
user32 = windll.user32


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
    capture = cap.init_camera()     # カメラ初期設定
    time_s = time.time()        # 処理時間
    time_sum = 0                # 終了処理関係
    # swing = Swing()             # スイング管理
    ball = Ball()               # ボール管理

    # フラグ
    start_flag = True          # 最初のジャッジが行われるまで開始しない
    detect_flag = 0         # ボールの検出を行うか
    judge_pre = "None"
    judge_curr = "None"

    # while(time_sum < 5):
    while(True):
        img_cap, img_main, img_bin = cap.prepare_img(capture)    # 画像を用意
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

        # detect_flag の管理
        if judge_pre != "None" and judge_curr == "None":
            detect_flag = 1
        if detect_flag == 1 and ball.tmp_pos[1] != (0, 0):
            detect_flag = 2
        if detect_flag == 2 and ball.tmp_pos[1] == (0, 0):
            detect_flag = 0

        # ボール位置を保存
        if detect_flag == 2:
            g_ball.append(ball.tmp_pos[1][1])
            ball.append()
        else:
            for i in ball.pos:
                print(i)
            ball.clear()

        #
        # ウィンドウ関連
        #
        # 処理時間
        time_e = time.time()
        freq = (int)(1.0 / (time_e - time_s))
        time_sum += time_e - time_s
        time_s = time_e
        g_f.append(freq)   # 描画用にデータを追加
        # キー入力
        hit_key = cv2.waitKey(1) & 0xFF
        if hit_key == ord('q'):
            break
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
        # cv2.circle(img_main, (100, 100), 10, pa.RED, 5)

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
# スイング
#
class Swing:
    def __init__(self):
        self.base_time = time.time()    # 基準にする時間
        self.wait_time = 0              # スイングするまで待機する時間
        self.is_clicking = False        # 左クリック中かどうか

    def process(self, ball):
        # スイングのタイミングを予約
        if self.is_clicking == False:
            if ball.len() > 0:
                if ball.pos[-1][1] > pa.POS_SPEED_MEASUREMENT_EDGE_Y:
                    if pa.TH_SWING_BALL_SPEED[0] <= ball.speed <= pa.TH_SWING_BALL_SPEED[1]:
                        self.wait_time = 1E-5
                        self.swing_pos = (ball.pos[-1][0], pa.POS_SWING_Y)
                        self.base_time = time.time()
        # クリック（DOWN）
        if self.wait_time != 0:
            if self.is_clicking == False:
                # if time.time() - self.base_time > self.wait_time:
                user32.mouse_event(pa.LEFT_DOWN, 0, 0, 0, 0)
                self.is_clicking = True
                self.base_time = time.time()
                self.wait_time = 0
        # クリック（UP）
        if self.is_clicking == True:
            if time.time() - self.base_time > pa.TIME_CLICKING:
                user32.mouse_event(pa.LEFT_UP, 0, 0, 0, 0)
                self.is_clicking = False


#
# ボール
#
class Ball:
    def __init__(self):
        self.pos = []       # ボール位置のリスト
        # 求めた位置を一時的に保管（0 : previous, 1 : current）
        self.tmp_pos = [(0, 0), (0, 0)]

    def clear(self):
        self.pos.clear()

    def append(self):
        self.pos.append(self.tmp_pos[1])  # ボール位置をリストに追加

    def len(self):
        return len(self.pos)

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


if __name__ == '__main__':
    main()
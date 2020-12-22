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
    graphWin = plot.PlotGraph()

    # 初期化処理
    capture = cap.init_camera()     # カメラ初期設定
    time_s = time.time()        # 処理時間
    time_sum = 0                # 終了処理関係
    # swing = Swing()             # スイング管理
    ball = Ball()               # ボール管理

    # フラグ
    start_flag = True          # 最初のジャッジが行われるまで開始しない
    detect_flag = True         # ボールの検出を行うか

    while(True):
        img_cap, img_main, img_bin = cap.prepare_img(capture)    # 画像を用意
        # swing.process(ball)                         # スイング処理
        judgement = judge.judge(img_main)           # 球のジャッジ

        # 最初のジャッジが行われるまで開始しない
        if judgement != "None":
            start_flag = True
        if start_flag == False:
            continue

        # detect_flag の管理
        # detect_flag = judgement == "None" and ball.is_throwing()
        # print(ball.is_throwing())

        ball.get_pos(img_bin)  # ボール位置を取得（main 座標系）
        if detect_flag == True:
            ball.append()          # ボール位置を追加
        else:
            ball.clear()

        # print(len(ball.pos))
        print(ball.tmp_pos)

        #
        # 30 秒経過で終了
        #
        # if time_sum > 30:
        #     break

        #
        # ウィンドウ関連
        #
        # 処理時間
        time_e = time.time()
        freq = (int)(1.0 / (time_e - time_s))
        time_sum += time_e - time_s
        time_s = time_e
        graphWin.append(freq)   # 描画用にデータを追加
        # キー入力
        hit_key = cv2.waitKey(1) & 0xFF
        if hit_key == ord('q'):
            break
        # ウィンドウに文字列表示
        cv2.rectangle(img_main, (5, 5), (150, 75), pa.WHITE, thickness=-1)
        cv2.putText(img_main, judgement, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, pa.RED, 2)
        # ボールの検出領域
        cv2.rectangle(img_main,
                      (pa.RECT_BALL_IN_MAIN[0], pa.RECT_BALL_IN_MAIN[1]),
                      (pa.RECT_BALL_IN_MAIN[0]+pa.RECT_BALL_IN_MAIN[2],
                       pa.RECT_BALL_IN_MAIN[1]+pa.RECT_BALL_IN_MAIN[3]),
                      pa.WHITE, thickness=2)
        # 検出したボールを表示
        cv2.circle(img_main, ball.tmp_pos, 10, pa.RED, 5)
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
        self.tmp_pos = ()   # 求めた位置を一時的に保管

    def clear(self):
        self.pos.clear()

    def append(self):
        if self.tmp_pos != (0, 0):
            self.pos.append(self.tmp_pos)  # ボール位置をリストに追加

    def len(self):
        return len(self.pos)

    #
    # ボール位置を求める（main 座標系で取得）
    #
    def get_pos(self, img_bin):
        ball_pos = (0, 0)
        mu = cv2.moments(img_bin, False)
        # 重心を計算
        if mu["m00"] != 0:
            ball_pos = (int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"]))
            ball_pos = tr.ball_main(ball_pos)  # ball -> main へ座標変換
        self.tmp_pos = ball_pos

    #
    # ボールが y+ 方向に移動しているか
    #
    def is_throwing(self):
        if len(self.pos) == 0:
            return True
        return self.tmp_pos[1] > self.pos[-1][1]


if __name__ == '__main__':
    main()

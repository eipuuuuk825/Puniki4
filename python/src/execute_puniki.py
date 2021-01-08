""" x, t のみを入力する場合には非対応 """

# 自分の
import transform as tr
import param as pa
import judge
import plot
import capture as cap
import utility
import so   # NN
# 自分のじゃないの
import cv2
import numpy as np
import time
import math
import sys
import datetime  # 現在時刻取得
# windows API
from ctypes import *
user32 = windll.user32

#
# パラメータ
#
path_learned_param = "data/param.csv"
path_data_set = "data/data_set.csv"     # 平均，標準偏差を利用する
path_raw_data_output = "data/"          # 検出したボールのデータの出力先
is_output_raw_data = False              # 検出したボールのデータを出力するか
time_clicking = 1                       # 左クリックを保持する時間
home_pos_x = 380                        # main 座標系におけるホームポジションのカーソル位置
offset_swing_time = -0.25               # スイングするタイミングのオフセット
offset_pos_tip = -15                    # バットの先端位置のオフセット


#
# ボールのデータを正規化して NN に渡せるようにする
#
def normalize(ball_data, mean, std):
    input_array = np.array(ball_data)
    normalize_data = input_array[:, 0:1].T
    normalize_data = np.append(normalize_data, input_array[:, 1:2].T, axis=1)
    normalize_data = np.append(normalize_data, input_array[:, 2:3].T, axis=1)
    mean_data = np.array(mean)[:-2].reshape(1, len(mean)-2)
    std_data = np.array(std)[:-2].reshape(1, len(std)-2)
    return ((normalize_data - mean_data) / std_data).ravel().tolist()


#
# 正規化前の値に戻す
#
def unnormalize(predict, mean, std):
    return [predict[0]*std[-2]+mean[-2], predict[1]*std[-1]+mean[-1]]


def main():
    #
    # NN の初期化
    #
    nn = so.NeuralNetwork("regression", path_learned_param)
    with open(path_data_set) as fileobj:
        # yt
        fileobj.readline()
        yt = float(fileobj.readline().split(",")[0])
        # mean
        fileobj.readline()
        mean = [float(val) for val in fileobj.readline().split(
            ",") if utility.is_num(val)]
        # std
        fileobj.readline()
        std = [float(val) for val in fileobj.readline().split(
            ",") if utility.is_num(val)]
    input_data_dim = len(mean)-2  # NN に入力するデータの次元

    #
    # グラフ
    #
    g_f = plot.PlotGraph('program frequency',
                         u'動作周波数 [Hz]',
                         max_y=60,
                         size=(500, 200),
                         pos=(0, 630))
    g_ball = plot.PlotGraph('ball y',
                            u'y [px]',
                            max_y=-1,
                            size=(500, 200),
                            pos=(500, 630))

    #
    # 初期化処理
    #
    capture = cap.init_camera()         # カメラ初期設定
    time_s = time.time()                # 処理時間
    time_sum = 0                        # 終了処理関係
    time_throw = time.time()            # ボールが投げられてから経過した時間
    time_judge_change = time.time()     # judge が変化してからの経過時間
    puniki = Puniki()                   # プニキ（移動，スイング）
    ball = Ball()                       # ボール管理
    predict = (0, 0, 0)                 # 予測（xt, yt, tt）
    home_pos_main = (int(home_pos_x), int(yt))    # ホームポジション

    # フラグ
    start_flag = False          # 最初のジャッジが行われるまで開始しない
    detect_flag = 0             # ボールの検出を行うか
    judge_pre = "None"
    judge_curr = "None"

    # while(time_sum < 100):
    while(True):
        # キー入力
        hit_key = cv2.waitKey(1) & 0xFF
        if hit_key == ord('q'):
            break

        # 画像を用意
        img_cap, img_main, img_bin_ball, img_bin_pu = cap.prepare_img(capture)

        # ジャッジ
        judge_pre = judge_curr              # judge を更新
        judge_curr = judge.judge(img_main)  # 球のジャッジ

        # 最初のジャッジが行われるまで開始しない
        if judge_curr != "None":
            start_flag = True
        if start_flag == False:
            continue

        # ボール位置を取得（main 座標系）
        ball.get_pos(img_bin_ball)

        # 終了
        if judge_curr == "Finish":
            ball.clear()
            detect_flag = 0
            start_flag = False
            continue

        # detect_flag の管理
        if judge_pre != "None" and judge_curr == "None":    # スタンバイOK
            detect_flag = 0.5
            predict = (0, 0, 0)  # 予測をリセット
            time_judge_change = time.time()
            puniki.has_swung = False
        if detect_flag == 0.5 and time.time() - time_judge_change > 0.5:
            detect_flag = 1
        if detect_flag == 1 and ball.tmp_pos[1] != (0, 0):  # 検出開始
            detect_flag = 2
            time_throw = time.time()  # ボールが投げられてからの時間を計測
        if detect_flag == 2 and ball.tmp_pos[1] == (0, 0):  # 検出終了
            detect_flag = 0
            # ball.output()
            ball.clear()
            utility.set_cursor(home_pos_main)    # カーソルをホームポジションに戻す

        # ボール位置を保存
        if detect_flag == 2:
            g_ball.append(ball.tmp_pos[1][1])   # グラフに描画
            ball.append(time.time() - time_throw)
            # NN で予測
            if len(ball.data)*3 == input_data_dim:
                input_data = normalize(ball.data, mean, std)
                ret = nn.compute(input_data)
                ret = unnormalize(ret, mean, std)
                predict = (int(ret[0]), int(yt), ret[1])

        #
        # プニキ
        #
        puniki.get_pos(img_bin_pu)  # 位置を求める
        if detect_flag != 0:
            if predict == (0, 0, 0):
                if home_pos_main[1] > puniki.pos_tip[1] + 10 and home_pos_main[1] > puniki.pos_tip[1]:
                    utility.set_cursor((home_pos_main[0], pa.RECT_MAIN[3] - 10))
                else:
                    utility.set_cursor(home_pos_main)
            else:
                puniki.set_pos((predict[0], predict[1]))
        puniki.swing(predict, time_throw)  # スイングする

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
        offset_str_pos_y = 180
        cv2.rectangle(img_main, (5, offset_str_pos_y), (130, offset_str_pos_y+60),
                      pa.WHITE, thickness=-1)
        cv2.putText(img_main, judge_curr,
                    (15, offset_str_pos_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, pa.RED, 1)
        cv2.putText(img_main, f'{time.time() - time_throw:.3f} [sec]',
                    (15, offset_str_pos_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, pa.RED, 1)
        cv2.putText(img_main, "detect flag : "+str(detect_flag),
                    (15, offset_str_pos_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, pa.RED, 1)

        # ボールの検出領域
        cv2.rectangle(img_main,
                      (pa.RECT_BALL_IN_MAIN[0], pa.RECT_BALL_IN_MAIN[1]),
                      (pa.RECT_BALL_IN_MAIN[0]+pa.RECT_BALL_IN_MAIN[2],
                       pa.RECT_BALL_IN_MAIN[1]+pa.RECT_BALL_IN_MAIN[3]),
                      pa.RED, thickness=1)
        # プニキの検出領域
        cv2.rectangle(img_main,
                      (pa.RECT_PUNIKI_IN_MAIN[0], pa.RECT_PUNIKI_IN_MAIN[1]),
                      (pa.RECT_PUNIKI_IN_MAIN[0]+pa.RECT_PUNIKI_IN_MAIN[2],
                       pa.RECT_PUNIKI_IN_MAIN[1]+pa.RECT_PUNIKI_IN_MAIN[3]),
                      pa.RED, thickness=1)
        # 検出したボールを表示
        if ball.tmp_pos[1] != (0, 0):
            cv2.circle(img_main, ball.tmp_pos[1], 1, pa.RED, 5)
        # 予測したボール位置を表示
        if predict != (0, 0, 0):
            cv2.circle(img_main, (predict[0], predict[1]), 9, pa.ORANGE, 2)
            if time.time() - time_throw >= predict[2]:
                cv2.circle(img_main, (predict[0], predict[1]), 15, pa.RED, 2)
        # 検出したプニキを表示
        cv2.circle(img_main, puniki.pos_puniki, 1, pa.BLUE, 5)
        # バットの先端を表示
        cv2.circle(img_main, puniki.pos_tip, 1, pa.BLUE, 5)

        # ウィンドウ表示
        cv2.imshow('capture', img_main)
        cv2.moveWindow('capture', 0, 0)
        cv2.imshow('binary', img_bin_ball)
        cv2.moveWindow('binary', 810, 0)
        cv2.imshow('puniki', img_bin_pu)
        cv2.moveWindow('puniki', 1230, 0)
    #
    # 終了処理
    #
    capture.release()
    cv2.destroyAllWindows()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


#
# プニキ
#
class Puniki:
    def __init__(self):
        self.pos_puniki = (0, 0)
        self.pos_tip = (0, 0)
        self.is_swinging = False            # スイング中かどうか
        self.has_swung = False
        self.time_swing_start = time.time()  # スイング開始時間

    # プニキとバット先端の位置を求める
    def get_pos(self, img_bin):
        # プニキ
        pos, flag = cap.calc_moment(img_bin)
        if flag:
            self.pos_puniki = tr.puniki_main(pos)
        # バット先端
        x = self.pos_puniki[0]+0.472*self.pos_puniki[1]+49.0+offset_pos_tip
        y = self.pos_puniki[1] + 10
        self.pos_tip = (int(x), int(y))

    # バット先端を目標位置に合わせる
    def set_pos(self, tip_tgt_pos):
        margin = 10
        pos_lu = (margin, margin)
        pos_ru = (pa.RECT_MAIN[2]-margin, margin)
        pos_ld = (margin, pa.RECT_MAIN[3]-margin)
        pos_rd = (pa.RECT_MAIN[2]-margin, pa.RECT_MAIN[3]-margin)

        cursor_pos = (0, 0)
        if self.pos_tip[0] > tip_tgt_pos[0]:
            if self.pos_tip[1] > tip_tgt_pos[1]:
                cursor_pos = pos_lu
            else:
                cursor_pos = pos_ld
        elif self.pos_tip[1] > tip_tgt_pos[1]:
            cursor_pos = pos_ru
        else:
            cursor_pos = pos_rd
        utility.set_cursor(cursor_pos)

    # スイングする
    def swing(self, predict, time_throw):
        if self.is_swinging:
            if time.time() - self.time_swing_start > 1:
                user32.mouse_event(pa.LEFT_UP, 0, 0, 0, 0)
                self.is_swinging = False
        else:
            if predict == (0, 0, 0) or self.has_swung:
                return
            if time.time() - time_throw >= predict[2] + offset_swing_time:
                user32.mouse_event(pa.LEFT_DOWN, 0, 0, 0, 0)
                self.time_swing_start = time.time()
                self.is_swinging = True
                self.has_swung = True


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
        pos, flag = cap.calc_moment(img_bin)
        if flag:
            self.tmp_pos[1] = tr.ball_main(pos)  # ball -> main へ変換

    #
    # データを出力
    #
    def output(self):
        if not is_output_raw_data:
            return

        filename = str(datetime.datetime.now()).replace(":", ".")
        print(filename)
        file = path_raw_data_output + filename + ".csv"
        with open(file, "w") as fileobj:
            for i in self.data:
                fileobj.write(str(i[0])+", "+str(i[1])+", "+str(i[2])+"\n")


if __name__ == '__main__':
    main()

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
path_learned_param = "data/iikanji_param.csv"
path_data_set = "data/data_set.csv"     # 平均，標準偏差を利用する
path_raw_data_output = "data/"          # 検出したボールのデータの出力先
is_output_raw_data = False              # 検出したボールのデータを出力するか


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
    time_throw = time.time()    # ボールが投げられてから経過した時間
    # swing = Swing()             # スイング管理
    ball = Ball()               # ボール管理
    predict = (0, 0, 0)        # 予測（xt, yt, tt）

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
        img_cap, img_main, img_bin = cap.prepare_img(capture)

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
        if judge_pre != "None" and judge_curr == "None":    # スタンバイOK
            detect_flag = 1
            predict = (0, 0, 0)  # 予測をリセット
        if detect_flag == 1 and ball.tmp_pos[1] != (0, 0):  # 検出中
            detect_flag = 2
            time_throw = time.time()  # ボールが投げられてからの時間を計測
        if detect_flag == 2 and ball.tmp_pos[1] == (0, 0):  # 検出終了
            # ball.output()
            ball.clear()
            detect_flag = 0

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
        # ウィンドウ関連
        #
        # 処理時間
        time_e = time.time()
        freq = (int)(1.0 / (time_e - time_s))
        time_sum += time_e - time_s
        time_s = time_e
        g_f.append(freq)   # 描画用にデータを追加

        # ウィンドウに文字列表示
        cv2.rectangle(img_main, (5, 5), (200, 75), pa.WHITE, thickness=-1)
        cv2.putText(img_main, judge_curr, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, pa.RED, 2)
        cv2.putText(img_main, f'{time.time() - time_throw:.3f} [sec]', (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, pa.RED, 2)
        # ボールの検出領域
        cv2.rectangle(img_main,
                      (pa.RECT_BALL_IN_MAIN[0], pa.RECT_BALL_IN_MAIN[1]),
                      (pa.RECT_BALL_IN_MAIN[0]+pa.RECT_BALL_IN_MAIN[2],
                       pa.RECT_BALL_IN_MAIN[1]+pa.RECT_BALL_IN_MAIN[3]),
                      pa.WHITE, thickness=2)
        # 検出したボールを表示
        cv2.circle(img_main, ball.tmp_pos[1], 1, pa.RED, 5)
        # 予測したボール位置を表示
        cv2.circle(img_main, (predict[0], predict[1]), 1, pa.BLUE, 5)
        if time.time() - time_throw >= predict[2]:
            cv2.circle(img_main, (predict[0], predict[1]), 10, pa.BLUE, 1)


        # ウィンドウ表示
        cv2.imshow('capture', img_main)
        cv2.moveWindow('capture', 10, 10)
        cv2.imshow('binary', img_bin)
        cv2.moveWindow('binary', 840, 10)
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

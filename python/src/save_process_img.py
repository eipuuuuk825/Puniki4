# 自分の
import capture as cap
import param as pa
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


def main():
    # 初期化処理
    capture = cap.init_camera()  # カメラ初期設定

    #
    # 画像の用意
    #
    # ret : フレームの画像を読み込めたかどうかを返す
    ret, img_cap = capture.read()
    if ret == False:
        print("capture.read err")
        exit()

    # 切り抜き
    img_main = img_cap[pa.RECT_MAIN[1]: pa.RECT_MAIN[1]+pa.RECT_MAIN[3],
                       pa.RECT_MAIN[0]: pa.RECT_MAIN[0]+pa.RECT_MAIN[2]]
    img_ball = img_cap[pa.RECT_BALL[1]: pa.RECT_BALL[1]+pa.RECT_BALL[3],
                       pa.RECT_BALL[0]: pa.RECT_BALL[0]+pa.RECT_BALL[2]]
    img_puniki = img_cap[pa.RECT_PUNIKI[1]: pa.RECT_PUNIKI[1]+pa.RECT_PUNIKI[3],
                         pa.RECT_PUNIKI[0]: pa.RECT_PUNIKI[0]+pa.RECT_PUNIKI[2]]

    # HSV に変換
    img_hsv_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2HSV)
    img_hsv_puniki = cv2.cvtColor(img_puniki, cv2.COLOR_BGR2HSV)

    # 二値化 (h_min, s_min, v_min), (h_max, s_max, v_max)
    # ボール
    img_bin_ball = cv2.inRange(img_hsv_ball, np.array(
        [0, 0, 230]), np.array([255, 15, 255]))
    # プニキ
    img_bin_puniki = cv2.inRange(img_hsv_puniki, np.array(
        [0, 220, 220]), np.array([5, 255, 255]))

    #
    # ウィンドウ表示
    #
    # cv2.imshow('ball_raw', img_ball)
    # cv2.imshow('ball_hsv', img_hsv_ball)
    # cv2.imshow('ball_bin', img_bin_ball)
    # cv2.imwrite("data/ball_raw.png", img_ball)
    # cv2.imwrite("data/ball_hsv.png", img_hsv_ball)
    # cv2.imwrite("data/ball_bin.png", img_bin_ball)
    cv2.imshow('puniki_raw', img_puniki)
    cv2.imshow('puniki_hsv', img_hsv_puniki)
    cv2.imshow('puniki_bin', img_bin_puniki)
    cv2.imwrite("data/puniki_raw.png", img_puniki)
    cv2.imwrite("data/puniki_hsv.png", img_hsv_puniki)
    cv2.imwrite("data/puniki_bin.png", img_bin_puniki)

    # キー入力
    while True:
        hit_key = cv2.waitKey(1) & 0xFF
        if hit_key == ord('q'):
            break

    #
    # 終了処理
    #
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

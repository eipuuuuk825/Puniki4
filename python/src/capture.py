import param as pa
import cv2
import numpy as np


#
# カメラ初期化処理
#
def init_camera():
    # VideoCapture オブジェクトを取得
    capture = cv2.VideoCapture(1)
    # カメラの設定
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, pa.CAP_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, pa.CAP_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, 30)   # なぜか 60 を指定しても 60fps にならない
    return capture


#
# 画像の前処理
#
def prepare_img(capture):
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
    return img_cap, img_main, img_bin_ball, img_bin_puniki


#
# 入力された二値画像の重心を求める
#
def calc_moment(img_bin):
    mu = cv2.moments(img_bin, False)
    ret = (0, 0)
    valid = False

    if mu["m00"] != 0:
        ret = (int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"]))
        valid = True

    return ret, valid

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
    # HSV に変換
    img_hsv = cv2.cvtColor(img_ball, cv2.COLOR_BGR2HSV)
    # 二値化
    img_bin = cv2.inRange(img_hsv, np.array(
        [0, 0, 200]), np.array([255, 10, 255]))
    # (h_min, s_min, v_min), (h_max, s_max, v_max)
    return img_cap, img_main, img_bin
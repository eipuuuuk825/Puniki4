import pyautogui
import cv2
import capture as cap
import transform as trans
import param as pa

second_window_offset = (128, -768)

capture = cap.init_camera()     # カメラ初期設定
while True:
    # マウスカーソル位置
    x, y = pyautogui.position()
    pos = (x - second_window_offset[0], y - second_window_offset[1])
    pos = trans.raw_cap(pos)
    if pos[0] >= pa.CAP_WIDTH or pos[1] >= pa.CAP_HEIGHT:
        continue

    img_cap, img_main, img_bin = cap.prepare_img(capture)    # 画像を用意
    print(pos[0], pos[1], img_cap[pos[1]][pos[0]])
    # print(pos)
    # print(img_cap.shape)

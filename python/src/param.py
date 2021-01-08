#
# 画面サイズ
#
RAW_HEIGHT = 767
RAW_WIDTH = 1365
CAP_HEIGHT = 720
CAP_WIDTH = 1280
SECOND_WINDOW_OFFSET = (128, -768)

#
# cap に対する切り抜き領域（x, y, width, height）
#
RECT_MAIN = (237, 24, 806, 604)
RECT_BALL = (450, 141, 417, 370)

# main 座標系における RECT_BALL
RECT_BALL_IN_MAIN = (RECT_BALL[0]-RECT_MAIN[0],
                     RECT_BALL[1]-RECT_MAIN[1],
                     RECT_BALL[2],
                     RECT_BALL[3])

#
# パラメータ
#
POS_SWING_Y = 420                   # スイング位置の y 座標（main）
POS_SPEED_MEASUREMENT_EDGE_Y = 250  # 320  # 球速測定領域下端 y 座標（main）
TH_SWING_BALL_SPEED = (30, 500)    # スイングする球速の閾値（最低，最高）
TIME_CLICKING = 1                   # 左クリックを保持する時間

#
# mouse_event
#
LEFT_DOWN = 2
LEFT_UP = 4

#
# 色
#
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
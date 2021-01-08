#
# 画面サイズ
#
RAW_HEIGHT = 767
RAW_WIDTH = 1365
CAP_HEIGHT = 720
CAP_WIDTH = 1280
SECOND_DISPLAY_OFFSET = (118, -771)

#
# cap に対する切り抜き領域（x, y, width, height）
#
RECT_MAIN = (237, 24, 806, 604)
RECT_BALL = (450, 141, 417, 370)
RECT_PUNIKI = (310, 300, 270, 220)

# main 座標系における RECT_BALL
RECT_BALL_IN_MAIN = (RECT_BALL[0]-RECT_MAIN[0],
                     RECT_BALL[1]-RECT_MAIN[1],
                     RECT_BALL[2],
                     RECT_BALL[3])

# main 座標系における RECT_PUNIKI
RECT_PUNIKI_IN_MAIN = (RECT_PUNIKI[0]-RECT_MAIN[0],
                       RECT_PUNIKI[1]-RECT_MAIN[1],
                       RECT_PUNIKI[2],
                       RECT_PUNIKI[3])

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
ORANGE = (0, 152, 243)

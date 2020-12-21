import param as pa

# ball：ボールを検知する矩形領域の座標系
# main：ゲーム画面の座標系
# cap：ManyCam によってキャプチャした画面の座標系
# raw：ディスプレイ上の座標系（マウスカーソルの位置指定に使用）

def ball_main(pos):
    xd = pos[0] + pa.RECT_BALL[2] - pa.RECT_MAIN[2]
    yd = pos[1] + pa.RECT_BALL[0] - pa.RECT_MAIN[0]
    return (xd, yd)


def main_cap(pos):
    xd = pos[0] + pa.RECT_MAIN[2]
    yd = pos[1] + pa.RECT_MAIN[0]
    return (xd, yd)


def cap_main(pos):
    xd = pos[0] - pa.RECT_MAIN[2]
    yd = pos[1] - pa.RECT_MAIN[0]
    return (xd, yd)


def cap_raw(pos):
    xd = pos[0] * pa.RAW_WIDTH // pa.CAP_WIDTH
    yd = pos[1] * pa.RAW_HEIGHT // pa.CAP_HEIGHT
    return (xd, yd)


def raw_cap(pos):
    xd = pos[0] * pa.CAP_WIDTH // pa.RAW_WIDTH
    yd = pos[1] * pa.CAP_HEIGHT // pa.RAW_HEIGHT
    return (xd, yd)

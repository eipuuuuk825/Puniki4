import param as pa
import transform as tr
# windows API
from ctypes import *
user32 = windll.user32

# 入力された文字列が数値を表すかを判定


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def set_cursor(pos_main):
    cursor_pos = tr.cap_raw(tr.main_cap(pos_main))
    cursor_x = cursor_pos[0] + pa.SECOND_DISPLAY_OFFSET[0]
    cursor_y = cursor_pos[1] + pa.SECOND_DISPLAY_OFFSET[1]
    user32.SetCursorPos(cursor_x, cursor_y)

# 入力された文字列が数値を表すかを判定
def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
import numpy as np

# x, y, R, G, B
# opencv のバージョンによってパラメータが違う？
# JUDGE_STRIKE = ((224, 310, 19, 95, 255),
#                 (283, 307, 21, 98, 255),
#                 (408, 320, 9, 88, 255),
#                 (496, 312, 18, 95, 255),
#                 (600, 312, 18, 95, 255)
#                 )
# JUDGE_HOMERUN = ((218, 243, 255, 130, 61),
#                  (297, 273, 255, 111, 33),
#                  (393, 246, 255, 129, 61),
#                  (466, 234, 255, 135, 68),
#                  (555, 234, 255, 135, 68))
# JUDGE_HIT = ((289, 267, 255, 182, 64),
#              (377, 285, 255, 176, 47),
#              (453, 265, 255, 182, 64))
# JUDGE_FOUL = ((254, 270, 61, 123, 255),
#               (329, 280, 51, 118, 255),
#               (458, 294, 36, 106, 255),
#               (538, 263, 66, 127, 255))
JUDGE_STRIKE = ((224, 310, 18, 94, 255),
                (283, 307, 22, 96, 255),
                (408, 320, 8, 87, 255),
                (496, 312, 18, 94, 255),
                (600, 312, 18, 94, 255)
                )
JUDGE_HOMERUN = ((218, 243, 255, 129, 60),
                 (297, 273, 255, 110, 32),
                 (393, 246, 255, 128, 59),
                 (466, 234, 255, 134, 67),
                 (555, 234, 255, 134, 67))
JUDGE_HIT = ((289, 267, 255, 180, 61),
             (377, 285, 255, 174, 46),
             (453, 265, 255, 180, 62))
JUDGE_FOUL = ((254, 270, 59, 123, 255),
              (329, 280, 50, 117, 255),
              (458, 294, 35, 105, 255),
              (538, 263, 65, 125, 255))


def judge(img_main):
    if judge_sub(img_main, JUDGE_STRIKE):
        return "Strike"
    elif judge_sub(img_main, JUDGE_HOMERUN):
        return "HomeRun"
    elif judge_sub(img_main, JUDGE_HIT):
        return "Hit"
    elif judge_sub(img_main, JUDGE_FOUL):
        return "Foul"
    else:
        return "None"


def judge_sub(img, param):
    for i in param:
        x = i[0]
        y = i[1]
        img_color = img[y][x]
        judge_color = np.array([i[4], i[3], i[2]])

        if not np.all(img_color == judge_color):
            return False
    return True

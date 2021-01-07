import pyautogui
import time
import cv2
import numpy


def main():
    img = numpy.empty((300, 300, 1))
    cv2.imshow("win", img)
    swing = Swing()

    while(True):
        swing.swing()
        # キー入力
        hit_key = cv2.waitKey(1) & 0xFF
        if hit_key == ord('q'):
            break
        elif hit_key == ord('h'):
            swing.swing(True)


class Swing:
    def __init__(self):
        self.down_time = time.time()
        self.is_down = False
        self.PUSHING_TIME = 1

    def swing(self, swing=False):
        if swing == True:
            pyautogui.mouseDown()
            self.down_time = time.time()
            self.is_down = True
            print("down")
        elif time.time() - self.down_time > self.PUSHING_TIME and self.is_down == True:
            pyautogui.mouseUp()
            self.is_down = False
            print("up")
        print(time.time() - self.down_time)


if __name__ == '__main__':
    main()

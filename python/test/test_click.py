from ctypes import *
import time

user32 = windll.user32

user32.mouse_event(2,0,0,0,0)
time.sleep(3)
user32.mouse_event(4,0,0,0,0)
import numpy as np
import cv2
import pyautogui
from PIL import Image
import time

last_time = time.time()

while(True):
    myScreenshot = pyautogui.screenshot()
    screen = np.array(myScreenshot.crop((0,0,660,460)))
    #screen = np.array(ImageGrab.grab(bbox=(0, 0, 660, 460)))
    cv2.imshow('Trial', screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    print(f'Loop took {time.time() - last_time} seconds.')
    last_time = time.time()

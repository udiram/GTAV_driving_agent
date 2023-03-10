import numpy as np
from PIL import ImageGrab
import cv2
import time



def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        screen = np.array(ImageGrab.grab(bbox=(3840-1600, 0, 3840, 900)))
        # print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', screen)
        cv2.resizeWindow('window', 1600,900)
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
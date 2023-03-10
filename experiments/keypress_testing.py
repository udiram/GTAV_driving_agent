import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from controls import PressKey, W, A, S, D


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img


def main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    while True:
        PressKey(W)
        screen = np.array(ImageGrab.grab(bbox=(3840-1600, 0, 3840, 900)))
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()
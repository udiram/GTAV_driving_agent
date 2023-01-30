import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from controls import PressKey, W, A, S, D


def canny(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=300, threshold2=400)
    return processed_img

def field_of_view(image):
    fov_img = image[300:, :]
    return fov_img

def process_img(image):
    fov_img = field_of_view(image)
    processed_img = canny(fov_img)
    return processed_img


def main():
    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    while True:
        # PressKey(W)
        screen = np.array(ImageGrab.grab(bbox=(3840-1600, 0, 3840, 900)))
        # screen_fov = field_of_view(screen)
        processed = process_img(screen)
        cv2.imshow('window', processed)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()
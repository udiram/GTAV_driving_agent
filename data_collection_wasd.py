import time
import cv2
import numpy as np
import pyautogui
import keyboard
from tqdm import tqdm

frames = []
keys_pressed = []

# Collect input training_1000
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)

for i in tqdm(range(1500)):
    print(i)
    screenshot = pyautogui.screenshot()
    screenshot = screenshot.crop((3840-1600, 0, 3840, 900))
    screenshot = screenshot.resize((800, 500))
    cv2.imshow('window', np.array(screenshot))
    cv2.imwrite('data/validation_1500/val_{}.png'.format(i), np.array(screenshot))
    cv2.waitKey(1)
    frame = np.array(screenshot)
    frames.append(frame)
    # check if w, a, s, d is pressed and append as a vector [w_activated, s_activated, a_activated, d_activated]
    keys = [0, 0, 0, 0, 0]
    key = cv2.waitKey(25)
    if key == -1:
        keys[4] = 1
        keys[0] = 0
        keys[1] = 0
        keys[2] = 0
        keys[3] = 0
    if keyboard.is_pressed('w'):
        keys[0] = 1
        keys[4] = 0
    if keyboard.is_pressed('s'):
        keys[1] = 1
        keys[4] = 0
    if keyboard.is_pressed('a'):
        keys[2] = 1
        keys[4] = 0
    if keyboard.is_pressed('d'):
        keys[3] = 1
        keys[4] = 0
    keys_pressed.append(keys)
    time.sleep(0.01)
    print(keys)
    i += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
#save training_1000
print(keys_pressed)
np.save('data/y_val.npy', keys_pressed)
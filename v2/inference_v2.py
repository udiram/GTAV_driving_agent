import cv2, time, random
import numpy as np
from utils.screen_grab import grab_screen
from utils.getkeys import key_check
from collections import deque, Counter
from models.models import inception_v3 as googlenet
from utils.direct_keys import PressKey, ReleaseKey, W, A, S, D
from statistics import mode, mean
from utils.motion_detection import motion_detection

GAME_WIDTH = 1920
GAME_HEIGHT = 1080

how_far_remove = 800
rs = (20, 15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen = log_len)

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen = 5)
hl_hist = 250
choice_hist = deque([], maxlen = hl_hist)

t_time = 0.25

model = googlenet(WIDTH, HEIGHT, 3, LR, output = 2)
MODEL_NAME = 'googlenet_selfdrive'
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')

def accel():
    PressKey(W)
    time.sleep(0.1)
    ReleaseKey(W)

def decel():
    ReleaseKey(W)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(3840 - 1600, 50, 3840, 950))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH,HEIGHT))
    t_minus, t_now, t_plus = prev, prev, prev

    while(True):
        screen = grab_screen(region=(3840 - 1600, 50, 3840, 950))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        last_time = time.time()
        screen = cv2.resize(screen, (WIDTH,HEIGHT))
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        delta_count_last = motion_detection(t_minus, t_now, t_plus)

        t_minus = t_now
        t_now = t_plus
        t_plus = screen
        t_plus = cv2.blur(t_plus, (4, 4))

        prediction = model.predict([screen.reshape(WIDTH,HEIGHT, 3)])[0]
        # prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])

        model_choice = np.argmax(prediction)
        print(model_choice, prediction)
        if model_choice == 0:
            for i in range(3):
                accel()
                choice_picked = 'accel'
        elif model_choice == 1:
            decel()
            choice_picked = 'no key'

        motion_log.append(delta_count_last)
        motion_avg = round(mean(motion_log), 3)
        print('loop took {0} seconds. Motion: {1}. Choice: {2}'.format(round(time.time() - last_time, 3),
                                                                           motion_avg, choice_picked))
        keys = key_check()
if __name__ == "__main__":
    main()
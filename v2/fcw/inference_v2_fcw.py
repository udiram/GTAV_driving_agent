import cv2, time, random
import numpy as np
from v2.utils.screen_grab import grab_screen
from v2.utils.getkeys import key_check
from collections import deque, Counter
from v2.models.models import inception_v3 as googlenet
from v2.utils.direct_keys import PressKey, ReleaseKey, W, A, S, D
from statistics import mode, mean
from v2.utils.motion_detection import motion_detection

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
MODEL_NAME = 'googlenet_selfdrivev1_fcw'
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')

def FCW():
    PressKey(S)
    time.sleep(0.1)
    ReleaseKey(S)

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
    color_select = (0,0,0)
    fcw = False
    while(True):
        if not paused:

            screen = grab_screen(region=(3840 - 1600, 50, 3840, 950))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            image = cv2.rectangle(screen,(0,0), (WIDTH, HEIGHT), color_select, 10)
            if fcw:
                cv2.putText(image, 'FCW', (WIDTH//2 - 50, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imwrite('fcw1.png', image)


            cv2.imshow('window1', image)
            # cv2.imshow('window', screen)
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
            #turn down fcw sensitivity
            prediction = np.array(prediction) * np.array([0.1, 1])
            model_choice = np.argmax(prediction)
            print(model_choice, prediction)
            if model_choice == 0:
                for i in range(5):
                    FCW()
                choice_picked = 'FCW'
                color_select = (0,0,255)
                fcw = True
                time.sleep(1)
            elif model_choice == 1:
                ReleaseKey(S)
                choice_picked = 'No FCW'
                color_select = (0,255,0)
                fcw = False

            motion_log.append(delta_count_last)
            motion_avg = round(mean(motion_log), 3)
            print('loop took {0} seconds. Motion: {1}. Choice: {2}'.format(round(time.time() - last_time, 3),
                                                                               motion_avg, choice_picked))
            keys = key_check()

            if 'T' in keys:
                if paused:
                    paused = False
                    print('Unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    time.sleep(1)
if __name__ == "__main__":
    main()
import os, time, cv2
import numpy as np
from utils.screen_grab import grab_screen
from utils.getkeys import key_check


key_map = {
    'W': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    'S': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    'A': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'WS': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'WD': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'SA': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'SD': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'NK': [0, 0, 0, 0, 0, 0, 0, 0, 1],
    # 'default': [0, 0, 0, 0, 0, 0, 0, 0, 1],
}


while True:
    file_name = 'data/training_data.npy'
    if os.path.isfile(file_name):
        print('File exists, moving along')
    else:
        print('File does not exist, starting fresh!')
        break

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    print(''.join(keys))
    if ''.join(keys) in key_map:
        return key_map[''.join(keys)]
    return key_map['NK']


def main(file_name):
    training_data = []
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    paused = False
    print('STARTING!!!')
    while(True):
        if not paused:
            print(len(training_data))
            screen = grab_screen(region = (3840-1600, 50, 3840, 950))
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480, 270))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
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
                np.save(file_name, np.array(training_data, dtype=object))
                print('SAVED')
                file_name = 'data/training_data.npy'
if __name__ == "__main__":
    main(file_name)
import time
from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab
from experiments.controls import PressKey, W, A, S, D

loaded_model = load_model('data/model.h5')

def predictions(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    return prediction

def control(prediction):
    if prediction[0][0] == 1:
        print('w')
        PressKey(W)
    if prediction[0][1] == 1:
        print('s')
        PressKey(S)
    if prediction[0][2] == 1:
        print('a')
        PressKey(A)
    if prediction[0][3] == 1:
        print('d')
        PressKey(D)

def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        screen = np.array(ImageGrab.grab(bbox=(3840-1600, 0, 3840, 900)))
        imS = cv2.resize(screen, (960, 540))  # Resize image
        print(predictions(screen))
        # control(predictions(screen))
        cv2.imshow('window', imS)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()
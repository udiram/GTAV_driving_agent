import time
from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab
from experiments.controls import PressKey, W, A, S, D, ReleaseKey
import pytesseract
import mss

loaded_model = load_model('data/model.h5')


def predictions(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    return prediction


def control(prediction, current):
    if abs(prediction - current) > 10:
        if prediction > current:
            print('w')
            PressKey(W)
            ReleaseKey(S)
            time.sleep(2)
            ReleaseKey(W)
        if prediction < current:
            print('no change')
            ReleaseKey(W)
            ReleaseKey(S)



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def get_speed(image):
    ROI = image[850:950, 675:950]
    speed = text_extraction(ROI)
    if len(speed) < 1:
        return 0
    speed = speed.strip()
    speed = speed.encode('ascii', 'ignore').decode('ascii')
    speed_int = get_digits(speed)
    return speed_int


def get_digits(string):
    # Iterate over each character in the string
    numbers = []
    for char in string:
        if char.isdigit() or char == '.':
            numbers.append(char)

    # Join the numbers into a single string
    numbers_str = ''.join(numbers)

    # Convert the string to an integer
    try:
        int_value = float(numbers_str)
        return int_value
    except ValueError:
        int_value = 0
        return int_value
        print("velocity not found, setting to 0")


def text_extraction(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # process the image more to extract the white text from the black box
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    # Recognize the text using Tesseract OCR
    result = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
    return result

def main():
    last_time = time.time()
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {"top": 50, "left": 3840-1600, "width": 1600, "height": 900}

        while (True):
            screen = np.array(sct.grab(monitor))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
            imS = cv2.resize(screen, (960, 540))  # Resize image
            cur_speed = get_speed(screen)
            prediction = predictions(screen)[0][0] * 100
            print("current speed: ", cur_speed)
            print("prediction: ", prediction)
            # run controls every 5 seconds
            if time.time() - last_time > 0.1:
                last_time = time.time()
                control(prediction, cur_speed)
            cv2.imshow('window', imS)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

main()

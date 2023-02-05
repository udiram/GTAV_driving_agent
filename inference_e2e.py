import time
from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab
from experiments.controls import PressKey, W, A, S, D
import pytesseract

loaded_model = load_model('data/model.h5')


def predictions(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    return prediction


def control(prediction):
    current_speed = get_speed(prediction)
    print("current speed: ", current_speed)
    print("prediction: ", prediction)

def get_speed(image):
    ROI = image[850:950, 675:950]
    # cv2.imshow('window', ROI)
    # cv2.imshow('window',cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB))
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
    while (True):
        # 800x600 windowed mode
        screen = np.array(ImageGrab.grab(bbox=(3840 - 1600, 0, 3840, 900)))
        imS = cv2.resize(screen, (960, 540))  # Resize image
        print(predictions(screen))
        # control(predictions(screen))
        cv2.imshow('window', imS)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()

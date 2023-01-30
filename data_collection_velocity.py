import numpy as np
from PIL import ImageGrab
import cv2
import time
import pytesseract
import re
from tqdm import tqdm


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
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

def screen_record():
    speeds = []
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    for i in tqdm(range(1000)):
        screen = np.array(ImageGrab.grab(bbox=(3840-1600, 50, 3840, 950)))
        # imS = cv2.resize(screen, (960, 540))  # Resize image
        frame = np.array(screen)
        #resize image
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite('data/img_res/img_{}.png'.format(i), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        speed = get_speed(screen)
        # print(speed)
        speeds.append(speed)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    np.save('data/y_img_res.npy', speeds)


screen_record()
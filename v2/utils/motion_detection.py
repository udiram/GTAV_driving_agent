import cv2

def delta_images(t0, t1, t2):
    d1 = cv2.absdiff(t2, t0)
    return d1

def motion_detection(t_minus, t_now, t_plus):
    delta_view = delta_images(t_minus, t_now, t_plus)
    _, delta_view = cv2.threshold(delta_view, 16, 255, cv2.THRESH_BINARY)
    delta_view = cv2.normalize(delta_view, None, 0, 255, cv2.NORM_MINMAX)
    img_count_view = cv2.cvtColor(delta_view, cv2.COLOR_RGB2GRAY)
    delta_count = cv2.countNonZero(img_count_view)
    delta_count_last = delta_count
    return delta_count
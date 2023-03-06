import cv2 as cv
import numpy as np
import math
# import time
# import matplotlib.pyplot as plt

global thresh_val
global thao
global initialized
global pixel_cm_ratio
pixel_cm_ratio = 1
initialized = False
thao_val = 20
thresh_val = 80


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def thresh_callback(val):
    global thresh_val
    thresh_val = val


def canny_callback(val):
    global thao_val
    thao_val = val


def init(to_init=True):
    global initialized
    initialized = to_init


def set_pixel_cm_ratio(ratio):
    global pixel_cm_ratio
    pixel_cm_ratio = ratio


if __name__ == '__main__':
    # set window name
    windows_name = "Object-measurement"
    cv.namedWindow(windows_name)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera detection failed")
    else:
        while True:
            # read configuration
            ret, frame = cap.read()

            # find contours
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            smoothed = cv.GaussianBlur(gray_frame, (13, 13), 0)
            ret, thresh = cv.threshold(smoothed, thresh_val, 255, 0)
            edged = cv.Canny(thresh, thao_val, thao_val*3)
            contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # get & draw approximated contours
            for cnt in contours:
                epsilon = 0.005 * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                # cv.drawContours(frame, [approx], 0, (0, 255, 0), 3)

            # get & draw rotated contours
                rect = cv.minAreaRect(approx)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                # cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # calculate size in pixels
                width_tmp = int(distance(box[0], box[1]))
                height_tmp = int(distance(box[0], box[3]))
                width = max(width_tmp, height_tmp)*pixel_cm_ratio
                height = min(width_tmp, height_tmp)*pixel_cm_ratio

            # get bounding box
                x, y, w, h = cv.boundingRect(cnt)

            # put text (write the object size)
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (x, y + h + 20)
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1
                frame = cv.putText(frame, "width: {} | height: {}".format(round(width, 3), round(height, 3)), org, font,
                                   fontScale, color, thickness, cv.LINE_AA)

            # draw bounding box
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # initialize real object size (with credit card)
            if not initialized:
                font_init = cv.FONT_HERSHEY_SIMPLEX
                org_init = (30, 30)
                fontScale_init = 0.5
                color_init = (0, 0, 255)
                thickness_init = 1
                cv.putText(frame, "place a standard credit card",
                           org_init, font_init, fontScale_init, color_init, thickness_init, cv.LINE_AA)

            # show the result
            cv.imshow(windows_name, frame)

            # get input from user
            key = cv.waitKey(1)
            if key == ord('q'):  # exit program (by pressing q on keyboard)
                break
            if key == ord('t'):  # reset global threshold (incase object wasn't detected)
                cv.createTrackbar("threshold", "Object-measurement", thresh_val, 255, thresh_callback)
            if key == ord('c'):  # reset global threshold (incase object wasn't detected)
                cv.createTrackbar("Canny Thao", "Object-measurement", thao_val, 200, canny_callback)
            if key == ord('i'):  # initialize pixel to cm ratio
                if len(contours) != 0:
                    credit_w_h_ratio = width / height if height else 1
                    if abs(credit_w_h_ratio - 1.585) < 0.1 and len(contours) != 0:
                        pixel_cm_ratio = 8.56 / width
                        set_pixel_cm_ratio(pixel_cm_ratio)
                        init()
            if key == ord('r'):
                init(False)
                set_pixel_cm_ratio(1)

    cap.release()
    cv.destroyAllWindows()

import numpy as np
import cv2 as cv
import pyautogui
import time
from mss import mss
from PIL import Image
import mediapipe as mp
import win32api
import win32con
from pynput import keyboard
import argparse
import sys
from sys import platform
import os

# Import Openpose as op (Windows)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    if platform == "win32":
        sys.path.append(dir_path + '/openpose-v1.7.0/build/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' \
                              + dir_path + '/openpose-v1.7.0/build/x64/Release;' \
                              +  dir_path + '/openpose-v1.7.0/build/bin;'
        import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. '
          'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()  # 1920x1080

Center_X = int(SCREEN_WIDTH/2)
Center_Y = int(SCREEN_HEIGHT/2)

Cx = Center_X * 65535 / SCREEN_WIDTH
Cy = Center_Y * 65535 / SCREEN_HEIGHT

Grab_Screen_Width = SCREEN_WIDTH*1/4
Grab_Screen_Height = SCREEN_HEIGHT*1/3

SCREEN_AREA_UPPER_LEFT_X = int(SCREEN_WIDTH*3/8)
SCREEN_AREA_UPPER_LEFT_Y = int(SCREEN_HEIGHT/3)
SCREEN_AREA_LOWER_RIGHT_X = int(SCREEN_WIDTH*5/8)
SCREEN_AREA_LOWER_RIGHT_Y = int(SCREEN_HEIGHT*2/3)

Compensate_X = SCREEN_WIDTH*3/8
Compensate_Y = SCREEN_HEIGHT/3

SCREEN_AREA_UPPER_LEFT = (SCREEN_AREA_UPPER_LEFT_X, SCREEN_AREA_UPPER_LEFT_Y)
SCREEN_AREA_LOWER_RIGHT = (SCREEN_AREA_LOWER_RIGHT_X, SCREEN_AREA_LOWER_RIGHT_Y)

AREA = {'top':SCREEN_AREA_UPPER_LEFT_Y,
        'left':SCREEN_AREA_UPPER_LEFT_X,
        'width':SCREEN_AREA_LOWER_RIGHT_X - SCREEN_AREA_UPPER_LEFT_X,
        'height':SCREEN_AREA_LOWER_RIGHT_Y - SCREEN_AREA_UPPER_LEFT_Y}


sct = mss()
screen = 'window'
cv.namedWindow(screen, 1)
cv.moveWindow(screen, 0, 0)

last_time = 0


params = dict()
params["model_folder"] = "openpose-v1.7.0/models/"
params["number_people_max"] = 1
params["model_pose"] = "COCO"

params["fps_max"] = 30
params["face"] = False
params["hand"] = False


# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


while True:

    last_time = time.time()

    image_mss = sct.grab(AREA)
    img_RGB = Image.frombytes('RGB', (image_mss.size.width, image_mss.size.height), image_mss.rgb)
    img_RGB = np.array(img_RGB)
    img_BGR = cv.cvtColor(img_RGB, cv.COLOR_BGR2RGB)

    # Process Image
    datum = op.Datum()
    datum.cvInputData = img_BGR
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    result = datum.cvOutputData


    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

    FPS = int(1 / (time.time() - last_time))
    cv.putText(result, str(FPS), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow(screen, result)

    if cv.waitKey(30) == ord('q'):
        cv.destroyAllWindows()
        break



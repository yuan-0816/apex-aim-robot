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

# Import Openpose (Windows)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
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


# from pynput.keyboard import Controller, Listener
# import threading

"""
　　　　　　      ／ ¯)
　　　　　　　  ／　／
　　　 　　  ／　／
　　　_／¯ ／　／'¯ )
　　／／ ／　／　／  ('＼
　（（ （　（　（　   ） )
　　＼　　　　　 ＼／  ／
　　　＼　　　　　　／
　　　　＼　　　　／
　　　　　＼　　　＼

　　　　　　 ＿＿＿
　　　　　／＞　　  フ
　　　　　|  　_　 _|
　 　　　／` ミ＿xノ
　　 　 /　　　 　 |
　　　 /　 ヽ　　 ﾉ
　 　 │　　|　|　|
　／￣|　　 |　|　|
　| (￣ヽ＿_ヽ_)__)
　＼二つ
"""

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


mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=0,
                   smooth_landmarks=True,
                   enable_segmentation=True,
                   smooth_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)

sct = mss()
screen = 'window'
cv.namedWindow(screen, 1)
cv.moveWindow(screen, 0, 0)

last_time = 0

paused = True
Mode = 1

def Image_to_Gray(img):
    image = img
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray

def on_press(key):
    global paused
    if paused:
        if "f3" in str(key):
            print("F3")
            paused = False
    else:
        if "f4" in str(key):
            print("F4")
            paused = True

def instruction():
    print("===============================")
    print("press 'f3' to start program")
    print("press 'f4' to stop program")
    print("press 'e' to aim the object")
    print("===============================")
    print("choice mode:")
    print("(1) Head")
    print("(2) Body")
    print("===============================")
    print("your mode?")
    while True:
        global Mode
        Mode = int(input())
        if Mode == 1:
            print("you choose Head mode!")
            break
        elif Mode == 2:
            print("you choose Body mode!")
            break
        else:
            print("choice your mode!!!")


def main():
    instruction()
    while True:
        key_listener = keyboard.Listener(on_press=on_press, suppress=False)
        key_listener.start()

        last_time = time.time()

        image_mss = sct.grab(AREA)

        img_RGB = Image.frombytes('RGB', (image_mss.size.width, image_mss.size.height), image_mss.rgb)
        img_RGB = np.array(img_RGB)
        img_BGR = cv.cvtColor(img_RGB, cv.COLOR_BGR2RGB)

        results = pose.process(img_BGR)

        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                # if id in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                if Mode == 1:
                    if id == 0:
                        cx, cy = int(lm.x * Grab_Screen_Width), int(lm.y * Grab_Screen_Height)
                        cv.circle(img_BGR, (cx, cy), 4, (0, 0, 255), cv.FILLED)
                        Target_X = cx + Compensate_X
                        Target_Y = cy + Compensate_Y
                        print("Head point is " + "x:" + str(Target_X) + " , " + "y:" + str(Target_Y))
                        nx = Target_X * 65535 / SCREEN_WIDTH
                        ny = Target_Y * 65535 / SCREEN_HEIGHT
                    if win32api.GetAsyncKeyState(0x02):
                    #     print("Aim the head!")
                        if paused == True:
                            win32api.mouse_event(win32con.MOUSEEVENTF_ABSOLUTE|win32con.MOUSEEVENTF_MOVE, int(nx), int(ny))
                        # time.sleep(0.3)
                elif Mode == 2:
                    # calculate the middle point in body
                    pass
        # show FPS
        FPS = int(1 / (time.time() - last_time))
        if paused == True:
            cv.putText(img_BGR, str(FPS), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv.putText(img_BGR, str(FPS), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow(screen, img_BGR)

        if cv.waitKey(30) == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    # main()
    pass
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
import tensorflow as tf


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

# Load model
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# interprinter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interprinter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
interprinter.allocate_tensors()


SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()  # 1920x1080

Center_X = int(SCREEN_WIDTH / 2)
Center_Y = int(SCREEN_HEIGHT / 2)

yuan = 1/12.5

Cx = Center_X * 65535 / SCREEN_WIDTH
Cy = Center_Y * 65535 / SCREEN_HEIGHT

Grab_Screen_Width = SCREEN_WIDTH * 1 / 4
Grab_Screen_Height = SCREEN_HEIGHT * 1 / 3

SCREEN_AREA_UPPER_LEFT_X = int(SCREEN_WIDTH * 3 / 8)
SCREEN_AREA_UPPER_LEFT_Y = int(SCREEN_HEIGHT / 3)
SCREEN_AREA_LOWER_RIGHT_X = int(SCREEN_WIDTH * 5 / 8)
SCREEN_AREA_LOWER_RIGHT_Y = int(SCREEN_HEIGHT * 2 / 3)

# Compensate_X = SCREEN_WIDTH * 3 / 8
# Compensate_Y = SCREEN_HEIGHT / 3

SCREEN_AREA_UPPER_LEFT = (SCREEN_AREA_UPPER_LEFT_X, SCREEN_AREA_UPPER_LEFT_Y)
SCREEN_AREA_LOWER_RIGHT = (SCREEN_AREA_LOWER_RIGHT_X, SCREEN_AREA_LOWER_RIGHT_Y)




Grab_Screen_Width = SCREEN_WIDTH*3/5
Grab_Screen_Height = SCREEN_HEIGHT*3/5
SCREEN_AREA_UPPER_LEFT_X = int(SCREEN_WIDTH/5)
SCREEN_AREA_UPPER_LEFT_Y = int(SCREEN_HEIGHT/5)
SCREEN_AREA_LOWER_RIGHT_X = int(SCREEN_WIDTH*4/5)
SCREEN_AREA_LOWER_RIGHT_Y = int(SCREEN_HEIGHT*4/5)
Compensate_X = SCREEN_WIDTH/5
Compensate_Y = SCREEN_HEIGHT/5






AREA = {'top': SCREEN_AREA_UPPER_LEFT_Y,
        'left': SCREEN_AREA_UPPER_LEFT_X,
        'width': SCREEN_AREA_LOWER_RIGHT_X - SCREEN_AREA_UPPER_LEFT_X,
        'height': SCREEN_AREA_LOWER_RIGHT_Y - SCREEN_AREA_UPPER_LEFT_Y}

sct = mss()
screen = 'window'
cv.namedWindow(screen, 1)
cv.moveWindow(screen, 0, 0)

last_time = 0
Mode = 1

def draw_keypoints(frame, keypoint, confindence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoint[0][0][0], [y, x, 1]))
    # for kp in shaped:
    #     ky, kx,  kp_conf = kp
    #     if kp_conf > confindence:
    #         cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
    ky, kx, kp_conf = shaped
    if kp_conf > confindence:
        cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
        target = (int(kx), int(ky))
        return target

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

def CompensetTarget(target):
    if target != None:
        (cx, cy) = target
        Target_X = cx + Compensate_X
        Target_Y = cy + Compensate_Y
        return Target_X, Target_Y
    else:
        return False

def main():
    instruction()
    while True:

        last_time = time.time()

        image_mss = sct.grab(AREA)
        img_RGB = Image.frombytes('RGB', (image_mss.size.width, image_mss.size.height), image_mss.rgb)
        img_RGB = np.array(img_RGB)
        img_BGR = cv.cvtColor(img_RGB, cv.COLOR_BGR2RGB)

        # Reshape Image
        img = img_BGR.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        # input_img = tf.cast(img, dtype=tf.uint8)
        input_img = tf.cast(img, dtype=tf.uint8)

        # Setup input and output
        input_details = interprinter.get_input_details()
        output_details = interprinter.get_output_details()

        # Make prediction
        interprinter.set_tensor(input_details[0]['index'], np.array(input_img))
        interprinter.invoke()
        keypoint_with_score = interprinter.get_tensor(output_details[0]['index'])

        target = CompensetTarget(draw_keypoints(img_BGR, keypoint_with_score, 0.3))
        if target:
            print("Head point is " + "x:" + str(target[0]) + " , " + "y:" + str(target[1]))
            nx = (target[0]-Center_X) * 65535 / SCREEN_WIDTH * yuan
            ny = (target[1]-Center_Y) * 65535 / SCREEN_HEIGHT * yuan
            if win32api.GetAsyncKeyState(0x06):
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(nx), int(ny))
                time.sleep(0.3)

        FPS = int(1 / (time.time() - last_time))
        cv.putText(img_BGR, str(FPS), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv.imshow(screen, img_BGR)

        if cv.waitKey(30) == ord('q'):
            cv.destroyAllWindows()
            break



if __name__ == "__main__":
    main()

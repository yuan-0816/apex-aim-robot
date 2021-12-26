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

yuan = 1/19

Cx = Center_X * 65535 / SCREEN_WIDTH
Cy = Center_Y * 65535 / SCREEN_HEIGHT

Grab_Screen_Width = SCREEN_WIDTH * 1 / 4
Grab_Screen_Height = SCREEN_HEIGHT * 1 / 3

SCREEN_AREA_UPPER_LEFT_X = int(SCREEN_WIDTH * 3 / 8)
SCREEN_AREA_UPPER_LEFT_Y = int(SCREEN_HEIGHT / 3)
SCREEN_AREA_LOWER_RIGHT_X = int(SCREEN_WIDTH * 5 / 8)
SCREEN_AREA_LOWER_RIGHT_Y = int(SCREEN_HEIGHT * 2 / 3)

Compensate_X = SCREEN_WIDTH * 3 / 8
Compensate_Y = SCREEN_HEIGHT / 3

SCREEN_AREA_UPPER_LEFT = (SCREEN_AREA_UPPER_LEFT_X, SCREEN_AREA_UPPER_LEFT_Y)
SCREEN_AREA_LOWER_RIGHT = (SCREEN_AREA_LOWER_RIGHT_X, SCREEN_AREA_LOWER_RIGHT_Y)



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
HeadMode = 1
BodyMode = 2

nose = 0
left_eye = 1
right_eye = 2
left_ear = 3
right_ear = 4
left_shoulder = 5
right_shoulder = 6
left_elbow = 7
right_elbow = 8
left_wrist = 9
right_wrist = 10
left_hip = 11
right_hip = 12
left_knee = 13
right_knee = 14
left_ankle = 15
right_ankle = 16


def draw_keypoints(frame, keypoint, confindence):
    y, x, c = frame.shape
    if Mode == HeadMode:
        shaped = np.squeeze(np.multiply(keypoint[0][0][0], [y, x, 1]))
        ky, kx, kp_conf = shaped
        if kp_conf > confindence:
            cv.circle(frame, (int(kx), int(ky)), 1, (0, 0, 255), 3)
            target = (int(kx), int(ky))
            return target
    elif Mode == BodyMode:
        shaped_left_shoulder = np.squeeze(np.multiply(keypoint[0][0][left_shoulder], [y, x, 1]))
        shaped_right_hip = np.squeeze(np.multiply(keypoint[0][0][right_hip], [y, x, 1]))
        left_shouldery, left_shoulderx, left_shoulder_conf = shaped_left_shoulder
        right_hipy, right_hipx, right_hip_conf = shaped_right_hip

        shaped_right_shoulder = np.squeeze(np.multiply(keypoint[0][0][right_shoulder], [y, x, 1]))
        shaped_left_hip = np.squeeze(np.multiply(keypoint[0][0][left_hip], [y, x, 1]))
        right_shouldery, right_shoulderx, right_shoulder_conf = shaped_right_shoulder
        left_hipy, left_hipx, left_hip_conf = shaped_left_hip

        if left_shoulder_conf > confindence and right_hip_conf > confindence:
            X = (left_shoulderx + right_hipx) / 2
            Y = (left_shouldery + right_hipy) / 2
            cv.circle(frame, (int(X), int(Y)), 1, (0, 0, 255), 3)
            target = (int(X), int(Y))
            return target
        elif right_shoulder_conf > confindence and left_hip_conf > confindence:
            X = (right_shoulderx + left_hipx) / 2
            Y = (right_shouldery + left_hipy) / 2
            cv.circle(frame, (int(X), int(Y)), 1, (0, 0, 255), 3)
            target = (int(X), int(Y))
            return target
        else:
            return None

def instruction():
    print("===============================")
    print("press 'mouse_left_click' to aim the object")
    print("press 'F4' to stop the cheater")
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
            print("choose your mode!!!")

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
            if win32api.GetAsyncKeyState(0x01):
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(nx), int(ny))
                # time.sleep(1)

        FPS = int(1 / (time.time() - last_time))
        cv.putText(img_BGR, str(FPS), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv.imshow(screen, img_BGR)
        cv.waitKey(30)
        if win32api.GetAsyncKeyState(0x73): #F4
            cv.destroyAllWindows()
            break



if __name__ == "__main__":
    main()

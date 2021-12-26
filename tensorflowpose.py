import tensorflow as tf
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time

last_time = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


interprinter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interprinter.allocate_tensors()

def draw_keypoints(frame, keypoint, confindence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoint, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confindence:
            cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


cap = cv.VideoCapture(2)
while cap.isOpened():

    last_time = time.time()

    ret , frame = cap.read()
    print(frame.shape)

    # Reshape Image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interprinter.get_input_details()
    output_details= interprinter.get_output_details()

    # Make prediction
    interprinter.set_tensor(input_details[0]['index'], np.array(input_img))
    interprinter.invoke()
    keypoint_with_score = interprinter.get_tensor(output_details[0]['index'])
    print(keypoint_with_score)

    # Rendering
    draw_keypoints(frame, keypoint_with_score, 0.4)

    FPS = int(1 / (time.time() - last_time))
    cv.putText(frame, str(FPS), (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('yuan', frame)

    if cv.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
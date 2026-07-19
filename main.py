import cv2
import mediapipe as mp
from mediapipe import ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import lmonimage

base_options: python.BaseOptions = python.BaseOptions(model_asset_path='hand_landmarker.task')
options: vision.HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

detector: vision.HandLandmarker = vision.HandLandmarker.create_from_options(options)
capture: cv2.VideoCapture = cv2.VideoCapture(0)

while capture.isOpened():
    frame: cv2.Mat
    success, frame = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    media_image = mp.Image(ImageFormat.SRGB, image)
    detected_image = lmonimage.draw_landmarks_on_image(image, detector.detect(media_image))
    image = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import lmonimage

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file('image.jpg')

detection = detector.detect(image)

annotated_image = lmonimage.draw_landmarks_on_image(image.numpy_view(), detection)
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2RGBA))
plt.axis('off')
plt.show()
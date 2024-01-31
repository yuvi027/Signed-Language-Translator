# Necessary library installations
# !pip install mediapipe opencv-python matplotlib tensorflow fiftyone
from IPython.display import clear_output
clear_output()

# Importing required libraries
import cv2
import math
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import load_img
import mediapipe as mp
import fiftyone as fo
import fiftyone.zoo as foz

# Constants
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Function to read and resize images
def read_n_resize(image_file, read=True):
    image = cv2.imread(image_file) if read else image_file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if read else image
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img

# MediaPipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Load a dataset
# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="train",
#     label_types=None,
#     classes=["person"],
#     max_samples=50,
# )

# # Pose detection on images from the dataset
# with mp_pose.Pose(
#         static_image_mode=True,
#         model_complexity=2,
#         enable_segmentation=True,
#         min_detection_confidence=0.5
# ) as pose:
#     for i, image_file in enumerate(dataset.view().take(10)):
#         image = read_n_resize(image_file.filepath)
#         image_height, image_width, _ = image.shape
#         results = pose.process(image)
#
#         if not results.pose_landmarks:
#             continue
#         print(
#             f'Nose coordinates: ('
#             f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#             f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#         )
#
#         annotated_image = image.copy()
#         mp_drawing.draw_landmarks(
#             annotated_image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#
#         plt.figure(figsize=(10, 10))
#         plt.imshow(annotated_image)
#         plt.show()

# MEDIAPIPE HANDS
# MediaPipe Hands Detection
mp_hands = mp.solutions.hands

# Load a dataset
# ... [Dataset loading code]

# Pose and Hand detection on images from the dataset
# with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
#                   min_detection_confidence=0.5) as pose, \
#         mp_hands.Hands(static_image_mode=True, max_num_hands=2,
#                        min_detection_confidence=0.5) as hands:
#     for i, image_file in enumerate(dataset.view().take(1)):
#         image = read_n_resize(image_file.filepath)
#         image_height, image_width, _ = image.shape
#
#         # Process for pose detection
#         pose_results = pose.process(image)
#
#         # Process for hand detection
#         hand_results = hands.process(image)
#
#         annotated_image = image.copy()
#
#         # Draw pose landmarks
#         if pose_results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 annotated_image,
#                 pose_results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#             )
#
#         # Draw hand landmarks
#         if hand_results.multi_hand_landmarks:
#             for hand_landmarks in hand_results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     annotated_image,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
#
#         plt.figure(figsize=(10, 10))
#         plt.imshow(annotated_image)
#         plt.show()


with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                  min_detection_confidence=0.5) as pose, \
        mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                       min_detection_confidence=0.5) as hands:
    # for i, image_file in enumerate(dataset.view().take(1)):
    image = read_n_resize("img.png")
    image_height, image_width, _ = image.shape

    # Process for pose detection
    pose_results = pose.process(image)

    # Process for hand detection
    hand_results = hands.process(image)

    annotated_image = image.copy()

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.show()

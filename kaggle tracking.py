from IPython.display import clear_output
# !pip install mediapipe pycocotools -q -U
clear_output()

# UTILITY
import cv2, math, os
import numpy as np
from random import shuffle, sample
from matplotlib import pyplot as plt
from tensorflow.keras.utils import load_img

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def read_n_resize(image_file, read=True):
    image = cv2.imread(image_file) if read else image_file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if read else image

    h, w = image.shape[:2]

    if h < w:
        img = cv2.resize(
            image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH)))
        )
    else:
        img = cv2.resize(
            image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT)
        )

    return img

# FACE DETECTION
import mediapipe as mp

# pick face detection solutions
mp_face_detection = mp.solutions.face_detection

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#SHORT RANGE SHOT
EXT = ('.jpg', '.png', '.jpeg')
SHORT_RANGE_INPUT_DIRECTORY = '../input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img'

spaths = sorted(
        [
            os.path.join(dirpath,filename)
            for dirpath, _, filenames in os.walk(SHORT_RANGE_INPUT_DIRECTORY)
            for filename in filenames if filename.endswith(EXT)
        ]
    )

shuffle(spaths)
len(spaths)
plt.figure(figsize=(10,10))
plt.imshow(load_img(spaths[1]))
plt.show()

with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5, model_selection=0
) as face_detection:
    for image_file in sample(spaths, 5):
        resized_image_array = read_n_resize(image_file)

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(resized_image_array)

        # Draw face detections of each face.
        print(f'Face detections of {image_file}:')
        if not results.detections:
            continue

        annotated_image = resized_image_array.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        resized_annotated_image = read_n_resize(annotated_image, read=False)

        plt.figure(figsize=(10, 10))
        plt.imshow(resized_annotated_image)
        plt.show()

#FULL RANGE SHOT
EXT = ('.jpg', '.png', '.jpeg')
FULL_RANGE_INPUT_DIRECTORY = '../input/aisegmentcom-matting-human-datasets/'

fpaths = sorted(
        [
            os.path.join(dirpath,filename)
            for dirpath, _, filenames in os.walk(FULL_RANGE_INPUT_DIRECTORY)
            for filename in filenames if filename.endswith(EXT)
        ]
    )

shuffle(fpaths)
len(fpaths)
with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5, model_selection=0
) as face_detection:
    for image_file in sample(fpaths, 5):
        resized_image_array = read_n_resize(image_file)

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(resized_image_array)

        # Draw face detections of each face.
        print(f'Face detections of {image_file}:')
        if not results.detections:
            continue

        annotated_image = resized_image_array.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        resized_annotated_image = read_n_resize(annotated_image, read=False)

        plt.figure(figsize=(10, 10))
        plt.imshow(resized_annotated_image)
        plt.show()

#MEDIAPIPE FACE MESH
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
) as face_mesh:
    for image_file in sample(fpaths, 5):
        resized_image = read_n_resize(image_file)
        results = face_mesh.process(resized_image)

        if not results.multi_face_landmarks:
            continue

        annotated_image = resized_image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style()
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image)
        plt.show()

from IPython.display import clear_output
# !pip install fiftyone
clear_output()

import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=None,
    classes=["person"],
    max_samples=50,
)

for sample in dataset.view().take(2):
    print(sample.filepath)
    plt.figure(figsize=(10,10))
    plt.imshow(load_img(sample.filepath))
    plt.show()

# MEDIAPIPE POSE
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
) as pose:
    for i, image_file in enumerate(dataset.view().take(10)):
        image = read_n_resize(image_file.filepath)

        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(image)

        if not results.pose_landmarks:
            continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()

        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image)
        plt.show()

# MEDIAPIPE HOLISTIC
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True
) as holistic:
    for image_file in dataset.view().take(1):
        image = read_n_resize(image_file.filepath)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = holistic.process(image)

        if results.pose_landmarks:
            print(
              f'Nose coordinates: ('
              f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
              f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            )

        annotated_image = image.copy()

        # Draw pose, left and right hands, and face landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())

        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image)
        plt.show()
import mediapipe as mp
import cv2
import os
import tensorflow as tf
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress unnecessary TensorFlow warnings
os.environ['TF_FORCE_GPU_PLACEMENT'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--xla_gpu_jit=true'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer.task',
delegate=mp.tasks.BaseOptions.Delegate.GPU)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#returns drawn on image and coords of the joints
def read_hands(image):
    with mp_hands.Hands(static_image_mode=False,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable= True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not results.multi_hand_landmarks:
            return image,None
        draw_landmarks(image,results.multi_hand_landmarks[0])
        return image,results
def draw_landmarks(image,landmark):
    mp_drawing.draw_landmarks(image, landmark, mp_hands.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
    return image
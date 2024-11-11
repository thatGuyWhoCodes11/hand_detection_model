from read_hand_landmarks import read_hands, draw_landmarks
import cv2
import numpy as np
import mediapipe as mp
import os

# Suppress unnecessary TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_PLACEMENT'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--xla_gpu_jit=true'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer.task',
                                    delegate=mp.tasks.BaseOptions.Delegate.GPU)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def add_fake_size(image, new_width, new_height):

    original_height, original_width, channels = image.shape

    new_height = new_height+original_height
    new_width = new_width+original_width
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    x_start = (new_width - original_width) // 2
    y_start = (new_height - original_height) // 2
    new_image[y_start:y_start + original_height,
              x_start:x_start + original_width] = image

    return new_image

# get opencv image and results of hand reading and crop accordingly


def read_and_crop(image):
    image = add_fake_size(image, 500, 500)
    h, w, c = image.shape
    drawn_image, results = read_hands(image)
    if not results:
        drawn_image = drawn_image[250:h-250, 250:w-250]
        return drawn_image, None
    higher_hand = get_higher_hand(results)
    if higher_hand["coords"][0].y > 0.5:
        drawn_image = drawn_image[250:h-250, 250:w-250]
        return drawn_image, None
    array_coords = np.array([[coord.x, coord.y]
                            for coord in higher_hand["coords"]])
    x = array_coords[:, 0]
    y = array_coords[:, 1]
    factor_max_x = np.max(x)
    factor_max_y = np.max(y)
    factor_min_x = np.min(x)
    factor_min_y = np.min(y)
    y_max_size = int(min(h, factor_max_y*h+200))
    x_max_size = int(min(w, factor_max_x*w+200))
    y_min_size = int(max(0, factor_min_y*h-200))
    x_min_size = int(max(0, factor_min_x*w-200))
    # print("{},{}".format(y_max_size, x_max_size))
    image = image[y_min_size:y_max_size, x_min_size:x_max_size]
    if higher_hand["handedness"] == "Left":
        image = cv2.flip(image, 1)
    _, results = read_hands(image)
    if not results:
        drawn_image = drawn_image[250:h-250, 250:w-250]
        return drawn_image, None
    coords = results.multi_hand_landmarks[0]
    # test_h = 50/h
    # test_w = 50/w
    # for coord in coords.landmark:
    #     coord.x = coord.x * test_h
    #     coord.y = coord.y * test_w
    # drawn_image=draw_landmarks(drawn_image,coords)
    drawn_image = drawn_image[250:h-250, 250:w-250]
    return drawn_image, coords


def get_higher_hand(hands):
    if len(hands.multi_hand_landmarks) == 1:
        return {"coords": hands.multi_hand_landmarks[0].landmark, "handedness": hands.multi_handedness[0].classification[0].label}
    hand0_height, hand1_height = hands.multi_hand_landmarks[
        0].landmark[0].y, hands.multi_hand_landmarks[1].landmark[0].y
    print("hand1:{}\nhand2:{}\nhand1Higher:{}".format(
        hand0_height, hand1_height, hand0_height > hand1_height))
    higher_hand = hands[0] if hand0_height > hand1_height else hands[1]
    return {"coords": higher_hand.multi_hand_landmarks[0].landmark, "handedness": higher_hand.multi_handedness[0].classification[0].label}

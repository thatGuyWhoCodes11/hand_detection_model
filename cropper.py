from read_hand_landmarks import read_hands , draw_landmarks
import cv2
import numpy as np
import mediapipe as mp
#get opencv image and results of hand reading and crop accordingly
def read_and_crop(image):
    drawn_image,results=read_hands(image)
    if not results:
        return drawn_image,None
    coords = results.multi_hand_landmarks[0]
    array_coords = np.array([[coord.x, coord.y] for coord in coords.landmark])
    x = array_coords[:, 0]
    y = array_coords[:, 1]
    factor_max_x = np.max(x)
    factor_max_y = np.max(y)
    factor_min_x = np.min(x)
    factor_min_y = np.min(y)
    h, w, c = image.shape
    y_max_size = int(min(h, factor_max_y*h+150))
    x_max_size = int(min(w, factor_max_x*w+150))
    y_min_size = int(max(0, factor_min_y*h-150))
    x_min_size = int(max(0, factor_min_x*w-150))
    print("{} {} {} {}".format(factor_max_y,h,factor_max_x,w))
    if 0 in (x_min_size,y_min_size) or h == y_max_size or w==x_max_size:
        return drawn_image,None
    # print("{},{}".format(y_max_size, x_max_size))
    image = image[y_min_size:y_max_size, x_min_size:x_max_size]
    drawn_image,results=read_hands(image)
    if not results:
        return drawn_image,None
    coords = results.multi_hand_landmarks[0]
    # test_h = 50/h
    # test_w = 50/w
    # for coord in coords.landmark:
    #     coord.x = coord.x * test_h
    #     coord.y = coord.y * test_w
    # drawn_image=draw_landmarks(drawn_image,coords)
    return drawn_image,coords
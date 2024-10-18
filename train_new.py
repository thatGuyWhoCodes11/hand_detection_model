from read_hand_landmarks import read_hands
import cv2
from csv_handler import write_to_csv
import time
from cropper import read_and_crop
cap = cv2.VideoCapture(0)

counter_limit = 100  # used to setup the number of frame images to be collected
letters = ["alef", "baA", "ta", "tha", "jeem", "ha", "kha","dal", "dhal", "ra","zay"]
letters = ["wow","yaA"]
for letter in letters:
    frame_counter = 0
    print("next letter is {}\nstarting in:".format(letter))
    for i in range(5, 1, -1):
        print(i)
        time.sleep(1)
    while cap.isOpened:
        _, image = cap.read()
        image,coords=read_and_crop(image)
        if not coords is None :
            frame_counter_str = str(frame_counter)
            frame_counter = frame_counter+1
            write_to_csv("new_coords", letter, coords.landmark)
            cv2.putText(image, 'saving frame: '+frame_counter_str, (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("camera", image)
        if frame_counter == counter_limit:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

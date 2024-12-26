import pickle
import cv2
from read_hand_landmarks import read_hands
import pandas as pd
import numpy as np
from cropper import read_and_crop
MODEL_NAME = "ArSL_model.pkl"
with open(MODEL_NAME, 'rb') as f:
    model = pickle.load(f)
cap=cv2.VideoCapture(0)
while cap.isOpened:
    ret,image=cap.read()
    drawn_image,coords=read_and_crop(image)
    if not coords is None:
        coords=coords.landmark
        #,coord.z removed
        coords_row = list(np.array([[coord.x,coord.y] for coord in coords]).flatten())
        X=pd.DataFrame([coords_row])
        hand_prediction = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        # print(str(model.predict(X))+"\n"+str(model.predict_proba(X)))
        cv2.putText(drawn_image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(drawn_image, hand_prediction.split(' ')[0] + " {}".format(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("test",drawn_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break
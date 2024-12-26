import pickle
import cv2
from read_hand_landmarks import read_hands
import pandas as pd
import numpy as np
from cropper import read_and_crop
import arabic_reshaper
from bidi.algorithm import get_display

def dictionary(letter):
     match letter:
          case "alef":
               return "ا"
          case "baA":
               return "ب"
          case "ta":
               return "ت"
          
def predict_hand(image,ARABIC_MODE):
    if ARABIC_MODE:
        MODEL_NAME = "ArSL_model.pkl"
    else:
        MODEL_NAME = "ASL_model.pkl"
    with open(MODEL_NAME, 'rb') as f:
        model = pickle.load(f)
    letters = []
    h,w,_=image.shape
    drawn_image,coords=read_and_crop(image)
    if not coords is None:
        coords=coords.landmark
    #,coord.z removed
        coords_row = list(np.array([[coord.x,coord.y] for coord in coords]).flatten())
        X=pd.DataFrame([coords_row])
        hand_prediction = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        prediction_proba=round(body_language_prob[np.argmax(body_language_prob)],2)
        prediction=hand_prediction.split(' ')[0]
        cv2.putText(drawn_image, 'CLASS'
                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.putText(drawn_image, prediction + " {}".format(prediction_proba)
                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return drawn_image,prediction,prediction_proba
    else:
        return drawn_image,0,0
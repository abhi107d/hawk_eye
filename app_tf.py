import tensorflow as tf
import cv2
import numpy as np
import time
from tensorflow.keras.models import Sequential
import mediapipe as mp
from tensorflow.keras.layers import LSTM, Dense
import os,sys

sys.path.insert(1, './utils/')
from frame_preprocessor import FramePreprocessor
from draw import Draw





label_map = [True, False]
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.load_weights(os.path.join("models","main1.h5"))   
  

mp_pos = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
frameprocessor=FramePreprocessor()
draw=Draw()
cam = cv2.VideoCapture('./videos_test/exercise.avi')
action = {}
text = ""
trsh = 0.6
res = np.array([0, 0])



with mp_pos.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hol:
    while cam.isOpened():
        ret,frame=cam.read()
        if not ret:
            break
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        landmarks=hol.process(image)

        #extracting trckobjects (persons)
        trackObjects=frameprocessor.extractTrackObjects(frame)

        for tob in trackObjects:
            if tob.extractedPoseLandmarks is not None:
                if tob.id not in action.keys():
                    action[tob.id]=[]
                else:
                    action[tob.id].append(tob.extractedPoseLandmarks)
                    action[tob.id]=action[tob.id][-30:]
                
                if len(action[tob.id])>=30:
                    res=model.predict(np.expand_dims(action[tob.id],axis=0))[0]
                    p_idx = np.argmax(res)
                    if res[p_idx]>trsh:
                        tob.predClass=label_map[p_idx]
        
        
        
         
        #drawing on image

        draw.drawTrack(trackObjects,frame)

        cv2.imshow("capture",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
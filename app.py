import tensorflow as tf
import cv2
import numpy as np
import time
from tensorflow.keras.models import Sequential
import mediapipe as mp
from tensorflow.keras.layers import LSTM, Dense
import os





def draw(frame,landmarks,mp_draw,mp_hol):
      
        mp_draw.draw_landmarks(frame,landmarks.pose_landmarks,mp_hol.POSE_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(98,226,34), thickness=2, circle_radius=2),
                              mp_draw.DrawingSpec(color=(238,38,211), thickness=2, circle_radius=2))
        
def extract_landmarks(landmarks):
    if landmarks.pose_landmarks:
        pose=np.array([[p.x,p.y,p.z,p.visibility] for p in landmarks.pose_landmarks.landmark]).flatten()
    else:
       
        return np.zeros(132)
    

    return np.concatenate([pose])


label_map=["cheating","not cheating"]

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.load_weights(os.path.join("models","main1.h5"))   
  

mp_pos=mp.solutions.pose
mp_draw=mp.solutions.drawing_utils
#just capture
cam=cv2.VideoCapture(0)
action=[]
text=[]
# predictions = []
trsh=0.6
res=np.array([0,0])

action=[]
label_map=["not cheating","cheating"]
text=" "
# predictions = []

trsh=0.9
res=np.array([0,0])

with mp_pos.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hol:
    while cam.isOpened():
        ret,frame=cam.read()
        if not ret:
            break
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        landmarks=hol.process(image)

        points=extract_landmarks(landmarks)

        #getting 30 frames of action
        action.append(points)
        action=action[-30:]
        if len(action)>=30:
            res=model.predict(np.expand_dims(action,axis=0))[0]
            # predictions.append(np.argmax(res))
            p_idx=np.argmax(res)
            # predictions=predictions[-10:]
            # print(predictions)
            # if np.unique(predictions)[-1]==np.argmax(res): 
               
            if  res[p_idx]>trsh:                             
                text=label_map[p_idx]
                          
              
           
        #drawing on image
        draw(frame,landmarks,mp_draw,mp_pos)
        frame=cv2.flip(frame,1)

        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame,text, (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("capture",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
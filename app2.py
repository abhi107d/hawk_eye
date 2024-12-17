import torch
import cv2
import numpy as np
import time
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import os, sys

sys.path.insert(1, './utils/')
from track import HumanTracker
from frame_preprocessor import FramePreprocessor
from draw import Draw

label_map = [True, False]

# Define the PyTorch model
class CheatingDetectionModel(nn.Module):
    def __init__(self):
        super(CheatingDetectionModel, self).__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(132, 64),
            nn.LSTM(64, 128),
            nn.LSTM(128, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.lstm(x)

# Load the model weights
model = CheatingDetectionModel()
#model.load_state_dict(torch.load(os.path.join("models", "main1.pth")))

mp_pos = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
frameprocessor=FramePreprocessor()
draw=Draw()
cam = cv2.VideoCapture(0)
action = {}
text = ""
trsh = 0.6
res = np.array([0, 0])

with mp_pos.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

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
                    input_data = torch.tensor(np.expand_dims(action[tob.id], axis=0), dtype=torch.float32)
                    res = model(input_data).detach().numpy()[0]
                    p_idx = np.argmax(res)
                    if res[p_idx]>trsh:
                        tob.predClass=label_map[p_idx]


        

        # Drawing on the image
        draw.drawTrack(trackObjects,frame)
        frame = cv2.flip(frame, 1)

        cv2.imshow("Hawk eye", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

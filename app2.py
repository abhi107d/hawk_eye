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

def draw(frame, landmarks, mp_draw, mp_hol):
    mp_draw.draw_landmarks(frame, landmarks.pose_landmarks, mp_hol.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(98, 226, 34), thickness=2, circle_radius=2),
                           mp_draw.DrawingSpec(color=(238, 38, 211), thickness=2, circle_radius=2))

def extract_landmarks(landmarks):
    if landmarks.pose_landmarks:
        pose = np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(132)
    return np.concatenate([pose])

label_map = ["cheating", "not cheating"]

# Define the PyTorch model
class CheatingDetectionModel(nn.Module):
    def __init__(self):
        super(CheatingDetectionModel, self).__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(132, 64, return_sequences=True),
            nn.LSTM(64, 128, return_sequences=True),
            nn.LSTM(128, 64, return_sequences=False),
            nn.Linear(64, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.lstm(x)

# Load the model weights
model = CheatingDetectionModel()
model.load_state_dict(torch.load(os.path.join("models", "main1.pth")))

mp_pos = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)
action = []
text = ""
trsh = 0.6
res = np.array([0, 0])

with mp_pos.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = hol.process(image)

        points = extract_landmarks(landmarks)

        # Getting 30 frames of action
        action.append(points)
        action = action[-30:]
        if len(action) >= 30:
            input_data = torch.tensor(np.expand_dims(action, axis=0), dtype=torch.float32)
            res = model(input_data).detach().numpy()[0]
            p_idx = np.argmax(res)
            if res[p_idx] > trsh:
                text = label_map[p_idx]

        # Drawing on the image
        draw(frame, landmarks, mp_draw, mp_pos)
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, text, (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("capture", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

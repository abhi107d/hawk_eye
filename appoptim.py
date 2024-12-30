import torch
import cv2
import numpy as np
import torch.nn as nn
import mediapipe as mp
import  sys

sys.path.insert(1, './utils/')
from track import HumanTracker
from frame_preprocessor import FramePreprocessor
from draw import Draw

label_map = [True, False]
class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,h1=None,h2=None,h3=None):
        x, h1 = self.lstm1(x,h1)
        x, h2 = self.lstm2(x,h2)
        x, h3 = self.lstm3(x,h3)
        x = self.relu(self.fc1(x))  # Use the last time step's output
        x = self.relu(self.fc2(x))
        #x = self.softmax(self.fc3(x))
        x = self.fc3(x)
        return x,h1,h2,h3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("models/lstm_model_full.pth")
model.to(device)  # Move to the appropriate device if needed
model.eval()  # Set the model to evaluation mode


mp_pos = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
frameprocessor=FramePreprocessor()
draw=Draw()
cam = cv2.VideoCapture(0)#"./videos_test/not_cheating.mp4")
action = {}
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
                    action[tob.id]={}   
                
                action[tob.id]['x']=tob.extractedPoseLandmarks
                action[tob.id]['h1']=None
                action[tob.id]['h2']=None
                action[tob.id]['h3']=None
                    
                
                if tob.id in action.keys():
                    
                    input_tensor = torch.tensor(np.array(action[tob.id]['x']), dtype=torch.float32)
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: shape becomes (1, 20, 132)
                    input_tensor = input_tensor.to(device)
                    with torch.no_grad():
                        output,h1,h2,h3 = model(input_tensor,action[tob.id]['h1'],action[tob.id]['h2'],action[tob.id]['h3'])  # Output shape: (1, num_classes)
                        prediction = torch.argmax(output, dim=1)
                        tob.predClass=label_map[prediction.item()] 
        # Drawing on the image
        draw.drawTrack(trackObjects,frame)
       
        cv2.imshow("Hawk eye", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

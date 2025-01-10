import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import sys
from collections import defaultdict,deque

sys.path.insert(1, './utils/')
from extract import Extractor
from draw import Draw

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

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.relu(self.fc1(x[:, -1, :]))  # Use the last time step's output
        x = self.relu(self.fc2(x))
        #x = self.softmax(self.fc3(x))
        x = self.fc3(x)
        return x
    


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = [True, False]

    model = torch.load("models/model.pth")
    model.to(device) 
    model.eval()  

    objmodel = YOLO("./weights/yolo11x-pose.pt").to(device)
    objmodel.eval()

    extractor=Extractor()
    draw=Draw()

    Mdict=defaultdict(lambda: deque(maxlen=seqlen))
    cam = cv2.VideoCapture("./videos_test/not_cheating.mp4")
    trsh = 0.6
    res = np.array([0, 0])
    seqlen=20

    while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            input,ididx,idclass=[],{},{}
            
            results=objmodel.track(frame, persist=True,tracker='bytetrack.yaml',verbose=False,device='cuda',conf=0.75)
            rslt=extractor.tensor(results)
            idx=results[0].boxes.id
            if idx is None or rslt is  None:
                continue
            idx=idx.int()
            for i in range(rslt.shape[0]):
                Mdict[idx[i].item()].append(rslt[i])
                if len(Mdict[idx[i].item()])>=seqlen:
                    input.append(torch.stack(list(Mdict[idx[i].item()])))
                    ididx[idx[i].item()]=len(input)-1
                
            if input:
                input=torch.stack(input).to(device)
                input=input.reshape(input.shape[0],input.shape[1],-1)      
                output = model(input) # Output shape: (N, num_classes)     
                prediction = torch.argmax(output, dim=-1)#(N)
                for k in ididx.keys():
                    idclass[k]=label_map[prediction[ididx[k]].item()]
            
            draw.drawBox(frame,results[0].boxes,idclass)
            #frame=results[0].plot()
            
            cv2.imshow("Hawk eye", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    run()
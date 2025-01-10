import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import sys
import argparse
from collections import defaultdict,deque

sys.path.insert(1, './utils/')
from extract import Extractor
from draw import Draw

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Program To Convert video into dataset")
    
    # Add required arguments
    parser.add_argument("--src",type=str,default=0, required=False, help="path")
    parser.add_argument("--modelpath",default="models/model.pth",type=str,required=False,help="model path")
    parser.add_argument("--posepath",default="./weights/yolo11x-pose.pt",type=str,required=False,help="pose model path")
    
    # Parse the arguments
    args = parser.parse_args()

    run=Run(args.src,args.modelpath,args.posepath)
    run.run()


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
    
class Run():
    def __init__(self,src,modelpath,weightpath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = [True, False]

        self.model = torch.load(modelpath)
        self.model.to(self.device) 
        self.model.eval()  

        self.objmodel = YOLO(weightpath).to(self.device)
        self.objmodel.eval()

        self.extractor=Extractor()
        self.draw=Draw()

        self.Mdict=defaultdict(lambda: deque(maxlen=self.seqlen))
        self.cam = cv2.VideoCapture(src)
        self.seqlen=20




    def run(self):

        while self.cam.isOpened():
                ret, frame = self.cam.read()
                if not ret:
                    break
                input,ididx,idclass=[],{},{}
                
                results=self.objmodel.track(frame, persist=True,tracker='bytetrack.yaml',verbose=False,device='cuda',conf=0.75)
                rslt=self.extractor.tensor(results)
                idx=results[0].boxes.id
                if idx is None or rslt is  None:
                    continue
                idx=idx.int()
                for i in range(rslt.shape[0]):
                    self.Mdict[idx[i].item()].append(rslt[i])
                    if len(self.Mdict[idx[i].item()])>=self.seqlen:
                        input.append(torch.stack(list(self.Mdict[idx[i].item()])))
                        ididx[idx[i].item()]=len(input)-1
                    
                if input:
                    input=torch.stack(input).to(self.device)
                    input=input.reshape(input.shape[0],input.shape[1],-1)      
                    output = self.model(input) # Output shape: (N, num_classes)     
                    prediction = torch.argmax(output, dim=-1)#(N)
                    for k in ididx.keys():
                        idclass[k]=self.label_map[prediction[ididx[k]].item()]
                
                self.draw.drawBox(frame,results[0].boxes,idclass)
                #frame=results[0].plot()
                
                cv2.imshow("Hawk eye", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        self.cam.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
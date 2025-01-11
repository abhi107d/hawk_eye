import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import argparse
from collections import defaultdict,deque
from src.utils import Extractor
from src.utils import Draw
from src.components.model import LSTMModel

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Program To Convert video into dataset")
    
    # Add required arguments
    parser.add_argument("--src",type=str,default=0, required=False, help="path")
    parser.add_argument("--modelpath",default="./models/model.pth",type=str,required=False,help="model path")
    parser.add_argument("--posepath",default="./weights/yolo11x-pose.pt",type=str,required=False,help="pose model path")
    
    # Parse the arguments
    args = parser.parse_args()

    run=Run(args.src,args.modelpath,args.posepath)
    run.run()


    
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

        self.dict=defaultdict(lambda: deque(maxlen=self.seqlen))
        self.cam = cv2.VideoCapture(src)
        self.seqlen=20




    def run(self):
  
        while self.cam.isOpened():
         
            ret, frame = self.cam.read()
            if not ret:
                break
            input,ididx,idclass=[],{},{}
            with torch.no_grad():
                results=self.objmodel.track(frame, persist=True,
                                            tracker='bytetrack.yaml'
                                            ,verbose=False,
                                            device='cuda',
                                            conf=0.75)
            rslt=self.extractor.tensor(results)
            idx=results[0].boxes.id
            if idx is None or rslt is  None:
                cv2.imshow("Hawk eye", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue
            idx=idx.int()
            for i in range(rslt.shape[0]):
                self.dict[idx[i].item()].append(rslt[i])
                if len(self.dict[idx[i].item()])>=self.seqlen:
                    input.append(torch.stack(list(self.dict[idx[i].item()])))
                    ididx[idx[i].item()]=len(input)-1
                
            if input:
                with torch.no_grad():
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
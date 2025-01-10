import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict

import argparse


from src.utils import Extractor


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Program To Convert video into dataset")
    
    # Add required arguments
    parser.add_argument("--videosrc",type=str, required=True, help="Path to the input video")
    parser.add_argument("--label",type=int,required=True,help="1 for cheating 2 for not cheating")
    parser.add_argument("--seqlen",type=int,required=False,default=20,help="seqlen of one datapoint in dataset")
    parser.add_argument("--show",type=bool,required=False,default=20,help="show detection")
    parser.add_argument("--posepath",type=str,required=False,default="weights/yolo11x-pose.pt",help="path to pose model")
    # Parse the arguments
    args = parser.parse_args()
    
                                
    #adjust values here
    labels=["cheating","non_cheating"]
    clas=labels[args.label-1]

    dc=DataCollector(args.videosrc,args.seqlen,args.posepath)
    dc.run(clas=clas,label=args.label,show=args.show)


class DataCollector:

    def __init__(self,sorce,seqlen,modelpath):
        with torch.no_grad():
            self.model = YOLO(modelpath).to('cuda')
            self.model.eval()

        # Open the video file
        self.seqlen=seqlen
        self.cap = cv2.VideoCapture(sorce)
        self.extractor=Extractor()
        self.dict = defaultdict(lambda: [])


    def show(self,frame,results):
                    # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints=results[0].keypoints.xy.int().cpu().tolist()
        # Visualize the results on the frame
        #frame= results[0].plot()
                        
        for (b,id,ky) in zip(boxes,track_ids,keypoints):
            cv2.putText(
                        frame,str(id),(int(b[0]), int(b[1])),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255, 0, 0),thickness=2                   
                )        
        for ky in keypoints:
            for k in ky:
                cv2.circle(frame,radius=2,center=(int(k[0]),int(k[1])),color=(255,255,33),thickness=5)
        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)
        # Break the loop if 'q' is pressed
        


    def run(self,clas,label,show=False):
        # Loop through the video frames
        
        Dataset=[]
        Yset=[]
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:                
                results = self.model.track(frame, persist=True,tracker='bytetrack.yaml',verbose=False,device='cuda',conf=0.75)
                ids=results[0].boxes.id
                if ids is None:
                    continue
                ids=ids.int()
                rslt=self.extractor.tensor(results)
                for i in range(rslt.shape[0]):
                  
                    self.dict[ids[i].item()].append(rslt[i])
                    if len(self.dict[ids[i].item()])==self.seqlen:          
                        Dataset.append(torch.stack(self.dict[ids[i].item()]))
                        Yset.append(label)
                        self.dict[ids[i].item()]=[]


                if show:
                    self.show(frame,results)
                    
                if cv2.waitKey(1) & 0xFF == ord("q") and show:
                    break

            else:
                # Break the loop if the end of the video is reached
                break
        Dataset_tensor = torch.stack(Dataset)
        Yset=torch.tensor(Yset)
        torch.save(Dataset_tensor, "Data/"+clas+'__x.pth')
        torch.save(Yset,"Data/"+clas+"__y.pth")
        print("SUCESS")
        self.cap.release()
        if show:
            cv2.destroyAllWindows()
  
     

if __name__ == "__main__":
    main()
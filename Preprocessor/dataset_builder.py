import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import sys
sys.path.insert(1, '../utils/')
from extract import Extractor



class DataCollector:

    def __init__(self,sorce,seqlen):
        with torch.no_grad():
            self.model = YOLO("../weights/yolo11x-pose.pt").to('cuda')
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
        torch.save(Dataset_tensor, "../Data/"+clas+'_Dataset.pth')
        torch.save(Yset,"../Data/"+clas+"_ylabel.pth")
        print("SUCESS")
        self.cap.release()
        if show:
            cv2.destroyAllWindows()


     
     
                                





                                
#adjust values here
labels=["cheating","non_cheating"]
no_frames=20
action=int(input("cheating=1 or non_cheating=2 : "))
clas=labels[action-1]

sorce="../videos_test/cheating.mp4"


dc=DataCollector(sorce,no_frames)
dc.run(clas=clas,label=action,show=False)




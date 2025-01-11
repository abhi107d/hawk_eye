import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
import pickle
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
    parser.add_argument("--database",type=str,required=False,default="Data/hawkeye.db",help="path to database")
    parser.add_argument("--buffersize",type=int,required=False,default=100,help="size of buffer to write")
    # Parse the arguments
    args = parser.parse_args()
    

    dc=DataCollector(args.videosrc,args.seqlen,args.posepath,args.database)
    dc.run(label=args.label-1,show=args.show)


class DataCollector:

    def __init__(self,sorce,seqlen,modelpath,database):
        try:
            with torch.no_grad():
                self.model = YOLO(modelpath).to('cuda')
                self.model.eval()
        except:
            print("Model not found")
            exit(0)
      
        self.seqlen=seqlen
        self.cap = cv2.VideoCapture(sorce)
        self.extractor=Extractor()
        self.dict = defaultdict(lambda: [])

        #database
        self.connection=sqlite3.connect(database)
        self.cursor=self.connection.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS data (id STRING PRIMARY KEY,pid INTEGER ,x BLOB, y INTEGER)")
        self.connection.commit()

        self.buffersize=100
        self.vidname=sorce.split('/')[-1].split(".")[0]


    def insert(self,dataset):
        try:
            self.cursor.executemany("INSERT OR IGNORE INTO data (id,pid,x,y) VALUES (?,?,?,?)",dataset)
            self.connection.commit()
            print("---------SAVED---------")
            
        except:
            print("---------SAVE FAILED---------")



    def run(self,label,show=False):
        # Loop through the video frames
        frameno=0
        Dataset=[]
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:                
                frameno+=1
                results = self.model.track(frame, persist=True,tracker='bytetrack.yaml',verbose=False,device='cuda',conf=0.75)
                ids=results[0].boxes.id
                if ids is None:
                    continue
                ids=ids.int()
                rslt=self.extractor.tensor(results)
                for i in range(rslt.shape[0]): 
                    id=ids[i].item() 
                    self.dict[id].append(rslt[i])
                    if len(self.dict[id])==self.seqlen:          
                        Dataset.append((str(frameno)+self.vidname+"_"+str(id),id,pickle.dumps(torch.stack(self.dict[id])),label))
                        self.dict[ids[i].item()]=[]

                if show:
                    frame= results[0].plot()
                    cv2.imshow("Hawk Eye", frame)
                      
                if cv2.waitKey(1) & 0xFF == ord("q") and show:
                    break
            else:
                break
            if len(Dataset)>self.buffersize:
                print("----SAVING {} SEQUENCES.....----".format(self.buffersize))
                self.insert(Dataset)
                Dataset=[]

        if len(Dataset)>0:
            self.insert(Dataset)

        self.connection.close()
        self.cap.release()
        if show:
            cv2.destroyAllWindows()
  
     

if __name__ == "__main__":
    main()
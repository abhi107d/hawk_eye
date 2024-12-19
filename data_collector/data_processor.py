import numpy as np
import cv2 
import os
import sys
import mediapipe as mp


sys.path.insert(1, '../utils/')
from track import HumanTracker
from draw import Draw
from frame_preprocessor import FramePreprocessor

class DataCollector:

    def __init__(self,path,sorce,videolength=None):  
        self.mp_hol=mp.solutions.pose
        self.mp_draw=mp.solutions.drawing_utils
        self.tracker=HumanTracker()
        self.draw=Draw()
        self.frameprocessor=FramePreprocessor()
        self.cam=cv2.VideoCapture(sorce)
        self.frameno={}
        self.videolength=videolength
        self.path=path
     
                                


    def saveCropAsImage(self,crops,frame):
        #for saving tracked persons images
        if self.videolength:
            for crop in crops:
                if crop.id not in self.frameno.keys():
                    self.frameno[crop.id]=[1,1]
                else:
                    self.frameno[crop.id][0]+=1   
                if self.frameno[crop.id][0]>self.videolength:
                    self.frameno[crop.id][0]=1
                    self.frameno[crop.id][1]+=1
                newpath=os.path.join(self.path,str(crop.id),
                                    str(self.frameno[crop.id][1]),
                                    )
                try:
                    os.makedirs(newpath)
                except:
                    pass
                image=frame[crop.y:crop.y+crop.h,crop.x:crop.x+crop.w].copy()
                cv2.imwrite(os.path.join(newpath,str(self.frameno[crop.id][0])+"_"+str(crop.objClass)+'.png'),image)
        else:
            for crop in crops:
                if crop.id not in self.frameno.keys():
                    self.frameno[crop.id]=1
                else:
                    self.frameno[crop.id]+=1          
                newpath=os.path.join(self.path,str(crop.id))
                try:
                    os.makedirs(newpath)
                except:
                    pass
                image=frame[crop.y:crop.y+crop.h,crop.x:crop.x+crop.w].copy()
                cv2.imwrite(os.path.join(newpath,str(self.frameno[crop.id])+"_"+str(crop.objClass)+".png"),image)

    def saveLandmarks(self,crops):
        #for saving tracked persons pose landmarks as np array
        if self.videolength:
            for crop in crops:
                if crop.id not in self.frameno.keys():
                    self.frameno[crop.id]=[1,1]
                else:
                    self.frameno[crop.id][0]+=1   
                if self.frameno[crop.id][0]>self.videolength:
                    self.frameno[crop.id][0]=1
                    self.frameno[crop.id][1]+=1
                newpath=os.path.join(self.path,str(crop.id),
                                    str(self.frameno[crop.id][1]),
                                    )
                try:
                    os.makedirs(newpath)
                except:
                    pass
                np.save(os.path.join(newpath,str(self.frameno[crop.id][0])+"_"+str(crop.objClass)+'.png'),crop.extractedPoseLandmarks)
        else:
            for crop in crops:
                if crop.id not in self.frameno.keys():
                    self.frameno[crop.id]=1
                else:
                    self.frameno[crop.id]+=1          
                newpath=os.path.join(self.path,str(crop.id))
                try:
                    os.makedirs(newpath)
                except:
                    pass
                np.save(os.path.join(newpath,str(self.frameno[crop.id])+"_"+str(crop.objClass)+".png"),crop.extractedPoseLandmarks)

    def run(self):
            ret=True        
            while ret:
            
                ret,frame=dc.cam.read()
                if not ret:
                    break

                #track and crope persons and create an array of TrackObjects     
                trackObjects=self.frameprocessor.extractTrackObjects(frame)

                if trackObjects:
                    #self.saveCropAsImage(trackObjects,frame)
                    # self.saveLandmarks(trackObjects)        
                    pass
                
                self.draw.drawTrack(trackObjects,frame,True) #for drawing               
                cv2.imshow("capture",frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):

                    break
            
            self.cam.release()
            cv2.destroyAllWindows()






                                
#adjust values here
labels=["cheating","non_cheating"]
no_frames=20
action=int(input("cheating=1 or non_cheating=2 : "))
action=labels[action-1]
_path=os.path.join("..\Data",action)
sorce='../videos_test/not_cheating.mp4'
# for vid in range(len(labels)):
#     try:
#         os.makedirs(os.path.join(_path))
#     except:
#         pass



dc=DataCollector(_path,sorce,no_frames)

dc.run()




import numpy as np
import cv2 
import os
import sys
import mediapipe as mp

sys.path.insert(1, '../tracker/')
from track import HumanTracker

class DataCollector:

    def __init__(self,path,sorce,videolength=None):  
        self.mp_hol=mp.solutions.pose
        self.mp_draw=mp.solutions.drawing_utils
        self.hol = self.mp_hol.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.tracker=HumanTracker()
        self.cam=cv2.VideoCapture(sorce)
        self.frameno={}
        self.videolength=videolength
        self.path=path
     

    def draw(self,frame,landmarks):
            
            self.mp_draw.draw_landmarks(frame,landmarks.pose_landmarks,self.mp_hol.POSE_CONNECTIONS,
                                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1))
                                
                                
    def extract_landmarks(self,landmarks):
        if landmarks.pose_landmarks:
            pose=np.array([[p.x,p.y,p.z,p.visibility] for p in landmarks.pose_landmarks.landmark]).flatten()
        else:
            return False
              
        return np.concatenate([pose])



    def cropOut(self,tracks,frame):
            crops=[]       
            if tracks:
                for i, track in enumerate(tracks):
                    if not track.is_confirmed() or track.get_det_class()!="person":
                        continue       
                    x, y, w, h = map(int, track.to_ltwh())
                    track_id = track.track_id
                    crop=frame[y:y+h, x:x+w]
                    if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0 and crop is not None:
                        crops.append((track_id,track.get_det_class(),crop))

                if crops:
                    return crops        
            return False

    def drawPosOnCrop(self,img):
        #draw landmarks on croped image and returns it
        image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        landmarks=self.hol.process(image)  
        self.draw(image,landmarks)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        return image

    def trackView(self,crops):
        #returns an array of croped images with 
        #drawn landmarks
        imgs=[] 
        for img in crops:
            image=self.drawPosOnCrop(img[2])  
            image=cv2.resize(image,(256,256))
            imgs.append(image)    
        # Concatenate all rows vertically
        return cv2.hconcat(imgs)


    def saveCropAsImage(self,crops):
        #for saving tracked persons images
        if self.videolength:
            for crop in crops:
                if crop[0] not in self.frameno.keys():
                    self.frameno[crop[0]]=[1,1]
                else:
                    self.frameno[crop[0]][0]+=1   
                if self.frameno[crop[0]][0]>self.videolength:
                    self.frameno[crop[0]][0]=1
                    self.frameno[crop[0]][1]+=1
                newpath=os.path.join(self.path,str(crop[0]),
                                    str(self.frameno[crop[0]][1]),
                                    )
                try:
                    os.makedirs(newpath)
                except:
                    pass
                cv2.imwrite(os.path.join(newpath,str(self.frameno[crop[0]][0])+"_"+str(crop[1])+'.png'),crop[2])
        else:
            for crop in crops:
                if crop[0] not in self.frameno.keys():
                    self.frameno[crop[0]]=1
                else:
                    self.frameno[crop[0]]+=1          
                newpath=os.path.join(self.path,str(crop[0]))
                try:
                    os.makedirs(newpath)
                except:
                    pass
                cv2.imwrite(os.path.join(newpath,str(self.frameno[crop[0]])+"_"+str(crop[1])+".png"),crop[2])

    def saveLandmarks(self,crops):
        #for saving tracked persons pose landmarks as np array
        if self.videolength:
            for crop in crops:
                if crop[0] not in self.frameno.keys():
                    self.frameno[crop[0]]=[1,1]
                else:
                    self.frameno[crop[0]][0]+=1   
                if self.frameno[crop[0]][0]>self.videolength:
                    self.frameno[crop[0]][0]=1
                    self.frameno[crop[0]][1]+=1
                newpath=os.path.join(self.path,str(crop[0]),
                                    str(self.frameno[crop[0]][1]),
                                    )
                try:
                    os.makedirs(newpath)
                except:
                    pass
                image=cv2.cvtColor(crop[2],cv2.COLOR_BGR2RGB)
                landmarks=self.hol.process(image) 
                conc_landmarks=self.extract_landmarks(landmarks)#to np array
                np.save(os.path.join(newpath,str(self.frameno[crop[0]][0])+"_"+str(crop[1])+'.png'),conc_landmarks)
        else:
            for crop in crops:
                if crop[0] not in self.frameno.keys():
                    self.frameno[crop[0]]=1
                else:
                    self.frameno[crop[0]]+=1          
                newpath=os.path.join(self.path,str(crop[0]))
                try:
                    os.makedirs(newpath)
                except:
                    pass
                image=cv2.cvtColor(crop[2],cv2.COLOR_BGR2RGB)
                landmarks=self.hol.process(image) 
                conc_landmarks=self.extract_landmarks(landmarks)#to np array
                np.save(os.path.join(newpath,str(self.frameno[crop[0]])+"_"+str(crop[1])+".png"),crop[2])

    def run(self):
            ret=True        
            while ret:
            
                ret,frame=dc.cam.read()
                if not ret:
                    break

                #track and crope persons
                tracks=self.tracker.track(frame)
                crops=self.cropOut(tracks,frame)

                if crops:
                    #dc.saveCropAsImage(crops)
                    #self.saveLandmarks(crops)        
                    # uncoment if realtime for viewing 
                    frame2=dc.trackView(crops)
                    cv2.imshow("capture2",frame2)
                
                self.tracker.drawTrack(tracks,frame,cv2) #for drawing               
                cv2.imshow("capture",frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):

                    break
            
            self.cam.release()
            cv2.destroyAllWindows()






                                
#adjust values here
labels=["cheating","non_cheating"]
no_frames=30
action=int(input("cheating=1 or non_cheating=2 : "))
action=labels[action-1]
_path=os.path.join("..\Data",action)
sorce='../videos_test/exercise.avi'
for vid in range(len(labels)):
    try:
        os.makedirs(os.path.join(_path))
    except:
        pass



dc=DataCollector(_path,sorce)

dc.run()




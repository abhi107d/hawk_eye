import numpy as np
import cv2 
import os
import sys
import mediapipe as mp

sys.path.insert(1, '../utils/')
from track import HumanTracker



#returns an array of this object after processing
class TrackObject:
    def __init__(self,id,w,h,x,y,objClass,poseLandmarks=None,predClass=False,extractedPoseLandmarks=None):
        self.id=id
        self.w=w
        self.h=h
        self.x=x
        self.y=y
        self.objClass=objClass
        self.poseLandmarks=poseLandmarks #pose object
        self.extractedPoseLandmarks=extractedPoseLandmarks #model input
        self.predClass=predClass #used to assign class



        

class FramePreprocessor:

    def __init__(self):  
        self.mp_hol=mp.solutions.pose
        self.mp_draw=mp.solutions.drawing_utils
        self.hol = self.mp_hol.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.tracker=HumanTracker()
       
                                   
                                
    def extractLandmarks(self,landmarks):
        if landmarks.pose_landmarks:
            pose=np.array([[p.x,p.y,p.z,p.visibility] for p in landmarks.pose_landmarks.landmark]).flatten()
        else:
            return np.zeros(132)
              
        return np.concatenate([pose])



    def processFrame(self,frame):
            #returns croped images after tracking
            tracks=self.tracker.track(frame)
            trackObject=[]       
            if tracks:
                for i, track in enumerate(tracks):
                    det_class=track.get_det_class()
                    if not track.is_confirmed():
                        continue       
                    x, y, w, h = map(int, track.to_ltwh())
                    track_id = track.track_id
                    #crop=frame[y:y+h, x:x+w]
                    #if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0 and crop is not None:
                    if x>=0 and y>=0 and x+w<=frame.shape[1] and y+h<=frame.shape[0]:
                        obj=TrackObject(id=track_id,w=w,h=h,x=x,y=y,objClass=det_class)
                        trackObject.append((obj))
                if trackObject:
                    return trackObject       
            return []

 

    def extractTrackObjects(self,frame):     
        trackObjects=self.processFrame(frame=frame)
        for tob in trackObjects:
            image=frame[tob.y:tob.y+tob.h,tob.x:tob.x+tob.w]
            if image is None:
                continue
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            landmarks=self.hol.process(image)
            tob.poseLandmarks=landmarks 
            extractedPoseLandmarks=self.extractLandmarks(landmarks)
            tob.extractedPoseLandmarks=extractedPoseLandmarks #model input
            #no need to store poseLandmarks if no drawing is neede
        return trackObjects
               

 

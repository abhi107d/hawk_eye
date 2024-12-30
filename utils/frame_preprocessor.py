import numpy as np
import cv2 
import os
import sys
import mediapipe as mp
from track import HumanTracker
from image_processor import ImageProcessor


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
        self.hol = self.mp_hol.Pose(static_image_mode=False,
        #smooth_landmarks=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ip=ImageProcessor()
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
                    #if x>=0 and y>=0 and x+w<=frame.shape[1] and y+h<=frame.shape[0]:
                    obj=TrackObject(id=track_id,w=w,h=h,x=x,y=y,objClass=det_class)
                    trackObject.append((obj))
                if trackObject:
                    return trackObject       
            return []
    
 

    def extractTrackObjects(self,frame):     
        trackObjects=self.processFrame(frame=frame)
        for tob in trackObjects:
            image=self.ip.preprocess_image(frame, tob, target_size=(256, 256))
            if image is None:
                continue
            landmarks=self.hol.process(image)
            tob.poseLandmarks=landmarks 
            extractedPoseLandmarks=self.extractLandmarks(landmarks)
            tob.extractedPoseLandmarks=extractedPoseLandmarks #model input
            #no need to store poseLandmarks if no drawing is neede
        return trackObjects
               

 

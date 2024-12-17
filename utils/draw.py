import cv2 
import mediapipe as mp


class Draw:
        
    def __init__(self):          
        self.mp_draw=mp.solutions.drawing_utils
        self.mp_hol=mp.solutions.pose
           
        

    def drawLandmarks(self,frame,landmarks):
        if not landmarks:
            return    
        self.mp_draw.draw_landmarks(frame,landmarks.pose_landmarks,self.mp_hol.POSE_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1),
                            self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1))
                                
                                
 

    def drawPoseOnCrop(self,trackObjects,frame):
        #returns an array of croped images with 
        #drawn landmarks
        imgs=[] 
        for tob in trackObjects:
            image=frame[tob.y:tob.y+tob.h,tob.x:tob.x+tob.w].copy() 
            self.drawLandmarks(image,tob.poseLandmarks)
            image=cv2.resize(image,(256,256))
            imgs.append(image)    
        # Concatenate all rows vertically
        return cv2.hconcat(imgs)
    

    
    def drawTrack(self,trackObjects,frame):
    #draw squares around tracks 
    
        for tob in trackObjects:
            # Draw bounding box and ID
            cv2.rectangle(frame, (int(tob.x), int(tob.y)), (int(tob.x + tob.w), int(tob.y + tob.h)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tob.id} | {tob.objClass}", (int(tob.x), int(tob.y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame
        



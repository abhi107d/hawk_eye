import cv2 



class Draw:
                           
    
    def drawBox(self,frame,boxes,predclass):
    #draw squares around tracks 
        xywhs=boxes.xywh
        ids=boxes.id.int()
        for xywh,id in zip(xywhs,ids):
            # Draw bounding box and ID
            color=(0,255,0)
            x,y,w,h=xywh
            w,h=w/2,h/2
            if id.item() in predclass.keys() and predclass[id.item()]:
                color=(0,0,255)
            cv2.rectangle(frame, (int(x-w), int(y-h)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(frame, f"ID: {id}", (int(x-w), int(y-h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame
        



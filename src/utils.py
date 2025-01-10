
import cv2 
import torch

class Extractor:
    def tensor(self,results):

        box=results[0].boxes.xywh
        pose=results[0].keypoints.xy
        conf=results[0].keypoints.conf
        if(box is None or pose is None or conf is None):
            return None
        box=box.to('cuda')
        pose=pose.to('cuda')

        #removing location info
        xy=box[:,:-2]
        xy = xy.unsqueeze(1).repeat(1, 17, 1)
        mask = (pose == 0).all(dim=-1, keepdim=True)
        rslt=pose-xy
        rslt=rslt * ~mask
        rslt[rslt==-0.000]=0.0

        #normalising
        wh=box[:,-2:]
        wh = wh.unsqueeze(1).repeat(1, 17, 1)
        rslt=rslt/wh

        #adding confidence
        conf = conf.unsqueeze(-1)  
        rslt = torch.cat((rslt, conf), dim=-1)
      
        return rslt



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
        
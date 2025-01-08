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

            


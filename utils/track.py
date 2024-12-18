from ultralytics import YOLOv10
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanTracker:

    def __init__(self):
        model_path = "../weights/yolov10x.pt"
        self.model = YOLOv10(model_path)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        logger.info(f"Using {device} as processing device")
        self.tracker = DeepSort(max_age=20, n_init=3) #,max_iou_distance=0.7)
        self.class_id=0
        self.conf=0.5



    def track(self,frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for det in results.boxes:
            for box in results[0].boxes:
                label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)     
                if class_id != self.class_id or confidence < self.conf:
                    continue
                
                detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, "person"])
            
        # Ensure detections are not empty before updating the tracker
        if len(detections) > 0:
            tracks = self.tracker.update_tracks(detections, frame=frame)
            return tracks
        
        return False
    

            

            

              


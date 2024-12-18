from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class HumanTracker:

    def __init__(self):
        self.model=YOLO('../yolov8n.pt')
        self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)



    def track(self,frame):


        # Run object detection
        results = self.model(frame,verbose=False)
        

        # Prepare detections for DeepSort
        detections = []
    
        for box in results[0].boxes:
            # Extract bounding box coordinates and confidence
            x1, y1, x2, y2 = map(float, box.xyxy[0])  # Ensure all values are Python floats
            conf = float(box.conf[0])
            class_id = int(box.cls[0].item())  # Class ID as integer
            class_name = self.model.names[class_id]  # Map class ID to class name

            # Append detection if confidence is above a threshold (e.g., 0.3)
            if conf > 0.8:
                detections.append(([x1, y1, x2-x1, y2-y1], conf,class_name))
            
        # Ensure detections are not empty before updating the tracker
        if len(detections) > 0:
            tracks = self.tracker.update_tracks(detections, frame=frame)
            return tracks
        
        return False
    

            

            

              


from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Initialize YOLO model

# Initialize DeepSort tracker

# Load video
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
    

    def drawTrack(self,tracks,frame,cv2):
            
            if tracks:
                for i, track in enumerate(tracks):
                    if not track.is_confirmed():
                        continue
                    # Extract track information
                    x, y, w, h = track.to_ltwh()
                    track_id = track.track_id

                

                    # Draw bounding box and ID
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id} | {track.get_det_class()}", (int(x), int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            

              


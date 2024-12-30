import cv2




class ImageProcessor:
    def crop(self,frame,tob):
        # Crop with padding
        padding = 20
        x1 = max(0, tob.x - padding)
        y1 = max(0, tob.y - padding)
        x2 = min(frame.shape[1], tob.x + tob.w + padding)
        y2 = min(frame.shape[0], tob.y + tob.h + padding)
        cropped_image = frame[y1:y2, x1:x2]
        
        # Ensure the cropped image is not empty
        if cropped_image is None or cropped_image.size == 0:
            return None
        return cropped_image
    

    def preprocess_image(self, frame, tob, target_size=(640, 480)):
        cropped_image = self.crop(frame, tob)
        if cropped_image is None:
            return None
        
        
        resized_image = cv2.resize(cropped_image, target_size)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
    
        preprocessed_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)
        
        return preprocessed_image

    

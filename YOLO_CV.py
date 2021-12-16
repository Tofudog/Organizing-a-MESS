# YOLO object detection
import cv2 as cv
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

# img_path = 'ImageDatabase/chess_masters.jpg'
# img = cv.imread(img_path)
# img = cv.resize(img, (960, 540))


class YOLO:
    
    def __init__(self, *args, **kwargs):
        self.weights = None
        self.net = None
        self.outputs = None
        self.layer_names = None
        self.start_time = time.time()
        
    def deviseModel(self, weights="Yolo_content/yolov3.weights"):
        self.weights = weights
        self.net = cv.dnn.readNetFromDarknet('Yolo_content/yolov3.cfg', weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        
        ln = self.net.getLayerNames()
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.layer_names = ln
        
      
    def retrain(self, img):
        # construct a blob from the image
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)     
        self.net.setInput(blob)
        self.outputs = self.net.forward(self.layer_names)
        
    def drawRectangles(self, img):
        boxes = []
        confidences = []
        classIDs = []
        h, w = img.shape[:2]
        
        for output in self.outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.9:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        ### Works with the following:
        # dimensions = np.array([w, h, w, h])
        # Make box values positive
        # box = detection[:, :, 0][0][:4] * dimensions
        
        # for box in boxes:
        #     x, y, w, h = box
        #     cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return boxes
    
    def kmeans(self):
        pass
 
def main():
    yolo_model = YOLO()
    yolo_model.deviseModel()
    start_time = yolo_model.start_time
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    time_interval = 0
    box = [0, 0, 0, 0]
    
    # So image rectangles do not get overriden
    # let us define all variables and continuously
    # draw rectangles that are same for given interval
    x, y, w, h = 0, 0, 0, 0
    rectangles = []
    
    while True:
        ret, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        # Get curr time to compare with start
        # so model is not being trained every ms
        curr_time = time.time()
        
        # You can info on image such as arr values
        image = np.zeros(frame.shape, np.uint8)
        image[:height, :width] = frame
        
        ### Unused unless faster YOLO or other net such as rCNN
        # for box in rectangles:
        #     x, y, w, h = box
        #     cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # yolo_model.retrain(image)
        #rectangles = yolo_model.drawRectangles(image)
        
        # Unused unless faster YOLO or other net
        def real_time_clf():
            global start_time, curr_time, time_interval
            # math ceil function may also work
            if int(curr_time - start_time) % 5 == 0:
                if int(curr_time - start_time) == time_interval:
                    pass  # continue if iterating
                else:
                    time_interval = int(curr_time - start_time)
                    yolo_model.retrain(image)
                    # Each rect is accessible via for loop
                    rectangles = yolo_model.drawRectangles(image)
                    
            return f"Boxes [x, y, w, h]: {rectangles}"


        cv.imshow('Show me the mess (press q)!', image)
        
        if cv.waitKey(1) == ord('q'):            
            break     
    
    cap.release()
    cv.destroyAllWindows()
    
    yolo_model.retrain(image)
    rectangles = yolo_model.drawRectangles(image)
    
    for box in rectangles:
         x, y, w, h = box
         cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv.imshow("Here is how you should clean the mess", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()









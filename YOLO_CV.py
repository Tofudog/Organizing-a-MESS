# YOLO object detection
import cv2 as cv
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans

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
        
     
    # Appropriately named for real-time clf (not just snapshot)
    def retrain(self, img):
        # construct a blob from the image
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)  
        print(f"blob shape: {blob.shape}")
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
                if confidence > 0.65:
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
    
    def organized_clusters(self, img, rectangles, n_groups=2):        
        model = KMeans(n_clusters=n_groups)
        
        # First get data on each rectangle
        found_objects = []
        cropped_img = None
        
        
        for data_point in rectangles:
            # Extract image given the data_point
            colLower, rowLower, colUpper, rowUpper = data_point
            colUpper += colLower
            rowUpper += rowLower
            cropped_img = img[rowLower:rowUpper, colLower: colUpper]
            
            # Must be of homogenous type for np array
            cropped_img = cv.dnn.blobFromImage(cropped_img, 1/255.0,
                                              (416, 416),
                                                swapRB=True, crop=False)
            cropped_img = cropped_img.reshape(416, 416, 3)[:, :, 0]
            found_objects.append(cropped_img.reshape(416*416,))
            
        
        # Flattening the image to have total ndim=2
        cluster_classes = model.fit_predict(found_objects)
        
        return found_objects, cluster_classes


 
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
        
# Alternative to real-time snapshot
def box_image(image_path):
    assert image_path is not None
    try:
        img = cv.imread(image_path)
        img = cv.resize(img, (960, 540))

    except FileNotFoundError:
        print("No such file exists")
    except Exception as e:
        print(type(e))
    else:
        yolo_model = YOLO()
        yolo_model.deviseModel()
        yolo_model.retrain(img)
        rectangles = yolo_model.drawRectangles(img)
        found, clusters = yolo_model.organized_clusters(img, rectangles)
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    colors = ["red", "blue", "yellow", "purple", "green"]
    
    # Create a rectangle patches
    for idx, box in enumerate(rectangles):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                  edgecolor=colors[clusters[idx]],
                                  facecolor='none')
        ax.add_patch(rect)
        
    plt.show()
    
    return found, clusters

    


# if __name__ == '__main__':
#     main()




# Needs reshaping functionality
def image_collage(images, clusters):
    n_images = len(images)
    n_rows = n_images
    n_cols = 1
    
    for N in range(1, n_images+1):
        if n_images % N == 0:
            if abs(n_rows - n_cols) > abs(n_images/N - N):
                n_rows, n_cols = int(n_images/N), int(N)
        
    
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    
    
    # Or reshaping list images
    currIdx = 0
    for row in range(n_rows):
        
        for col in range(n_cols):
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([clusters[currIdx]])
            axs[row, col].set_yticklabels([clusters[currIdx]], fontsize=45)  
            axs[row, col].imshow(images[currIdx].reshape(416, 416))
            currIdx += 1
    
        
    fig.set_size_inches(18.5, 10.5)
    fig.set_figwidth(25)
    fig.set_figheight(25)
    fig.savefig("ImageDataBase/cluster_collage.jpg", dpi=100)

    plt.show()


























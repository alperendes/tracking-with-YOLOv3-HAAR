import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import imshow

#LOAD YOLO
net = cv2.dnn.readNet("C:/Users/alper/Desktop/graduation project/yolo test/yolov3_whitespot.weights",
                      "C:/Users/alper/Desktop/graduation project/yolo test/yolov3_testing.cfg")
#SHOW CLASSES
classes = ["detected"]

layers = net.getLayerNames()
output_layers = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0 , 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN
#Video capturing
cap = cv2.VideoCapture('C:/Users/alper/Desktop/graduation project/videos/fish5_trial2_095hz_01cm.mov')
starting_time = time.time()
img_id = 0
a = 0
while True:
    ret, img = cap.read()
    img_id += 1
    height, width, channels = img.shape
    #Creating different colored blobs for easy detection, tracking
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
         for detection in out:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.5:
                   #Object detected
                   center_x = int(detection[0] * width)
                   center_y = int(detection[1] * height)
                   w = int(detection[2] * width)
                   h = int(detection[3] * height)
                   #Creating rectangle
                   x = int(center_x - w/2)
                   y = int(center_y - h/2)
                   boxes.append([x, y, w, h])
                   confidences.append(float(confidence))
                   class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, (255, 0, 255), 2)
            #Masking out the color
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            low_blue = np.array([94, 80, 2])
            high_blue = np.array([126, 255, 255])
            mask = cv2.inRange(hsv, low_blue, high_blue)
            blue = cv2.bitwise_and(img, img, mask=mask)
            contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #Calculating the center of the rectangle
            ave = np.average(contour[0][1],axis=0)
            x_comp = int(ave[0][0])
            column_names=["frame", "position"]
            if a == 0:
                df = pd.DataFrame([[a, x_comp]], columns = column_names)
                a += 1
            else:
                df_append = pd.DataFrame([[a, x_comp]],columns=column_names)
                df = df.append(df_append)
                a += 1
    #FPS settings
    elapsed_time = time.time() - starting_time
    fps = img_id / elapsed_time
    #Printing out important values
    cv2.putText(img, "Frame per Second: " +str(fps), (10, 60), font, 2, (255, 0, 255), 2)
    cv2.putText(img, "Confidence: " +str(confidences), (10, 30), font, 2, (255, 0, 255), 2)
    #Video
    cv2.imshow("LIVE", img)
    #cv2.imwrite("{:d}.jpg".format(a), img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#Plotting the nodal point
plot2 = plt.figure(2)
plt.plot(df['frame'], df['position'])
plot2.show()
plt.show()
#print(confidences)
cv2.destroyAllWindows()

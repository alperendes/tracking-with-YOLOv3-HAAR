import cv2
import numpy as np
import time

fish_cascade = cv2.CascadeClassifier('C:/Users/alper/Desktop/graduation project/HAAR/oneLastHaar/oneLastHaar.xml')
cap = cv2.VideoCapture('C:/Users/alper/Desktop/graduation project/VIDEOS/fish7_trial2_095hz_01cm.mov')
i = 0

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
img_id = 0
while True:
    ret, img = cap.read()
    img_id += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    head = fish_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in head:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
    elapsed_time = time.time() - starting_time
    fps = img_id / elapsed_time
    cv2.putText(img, "Frame per Second: " +str(fps), (10, 30), font, 2, (0, 0, 255), 2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30)& 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
 

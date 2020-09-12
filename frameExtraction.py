import cv2

vidcap = cv2.VideoCapture('C:/Users/alper/Desktop/graduation project/videos/fish7_trial5_205hz_01cm.mov')
count = 0
while vidcap.isOpened():
  ret, frame = vidcap.read()
  if ret:
    cv2.imwrite("frame4.2_%d.jpg" % count, frame)
    count += 30
    vidcap.set(1, count)
  else:
    vidcap.release()
  print(count)

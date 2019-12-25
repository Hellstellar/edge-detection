import numpy as np


import cv2

cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml');

imgPath = 'H:/edge-detection/kalatest.png';
img = cv2.imread(imgPath);
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

body = cascade.detectMultiScale(
    gray,
    scaleFactor = 1.001,
    minNeighbors = 5,
    minSize = (30,30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in body:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Upper Body',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
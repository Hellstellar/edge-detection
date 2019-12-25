import cv2
import numpy as np
import face_recognition


lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


image = cv2.imread('D:/dev/project-snow-white/edge detection/tests.jpg')

face_locations = []

height, width, channels = image.shape

rgb_image = image[:, :, ::-1]

face_locations = face_recognition.face_locations(rgb_image)

(_, _, bottom, _) = face_locations[0]

image = image[bottom:height, 0:width]

converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)

cv2.imshow('skin mask', skinMask)

# apply a series of erosions and dilations to the mask
# using an elliptical kernel
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# skinMask = cv2.erode(skinMask, kernel, iterations = 2)
# skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

# blur the mask to help remove noise, then apply the
# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(image, image, mask = skinMask)

final = cv2.subtract(image, skin)

cv2.imshow("images", np.hstack([image, final]))
print('hello')
cv2.waitKey(0)
cv2.destroyAllWindows()
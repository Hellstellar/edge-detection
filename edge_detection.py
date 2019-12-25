from __future__ import print_function
import cv2
import face_recognition
import numpy as np
import argparse
import random as rng

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


image = cv2.imread('H:/edge-detection/kalatest.png')
face_locations = []

height, width, channels = image.shape

rgb_image = image[:, :, ::-1]

face_locations = face_recognition.face_locations(rgb_image)

(_, _, bottom, _) = face_locations[0]

image = image[bottom:height, 0:width]

# Create a kernel that we will use to sharpen our image
# an approximation of second derivative, a quite strong kernel
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
# imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
# sharp = np.float32(src)
# imgResult = sharp - imgLaplacian
# # convert back to 8bits gray scale
# imgResult = np.clip(imgResult, 0, 255)
# imgResult = imgResult.astype('uint8')
# imgLaplacian = np.clip(imgLaplacian, 0, 255)
# imgLaplacian = np.uint8(imgLaplacian)
# #cv.imshow('Laplace Filtered Image', imgLaplacian)

converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)

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

# # Converting the image to grayscale.
gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

# Smoothing without removing edges.
gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

edges_filtered = cv2.Canny(gray_filtered, 50, 120)


conts, h = cv2.findContours(edges_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

c = max(conts, key = cv2.contourArea)


cv2.drawContours(gray , c, -1, (255,255,255), 2)

print(len(conts))

edges_filtered = edges_filtered[0:height//2, 0:width]

# Stacking the images to print them together for comparison
#images = np.hstack((gray, edges_filtered))

# Display the resulting frame
cv2.imshow('Frame', edges_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
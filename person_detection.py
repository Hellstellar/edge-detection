from torchvision import models
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
import sys
import cv2

np.set_printoptions(threshold=sys.maxsize)
def decode_segmap(image, nc=21):
  if np.any(image == 15):
	  label_colors = []
	  found_person = False
	  label_colors = np.array([(0, 0, 0) if idx != 15 else (255, 255, 255) for idx in range(0,nc)])
	  

	  r = np.zeros_like(image).astype(np.uint8)
	  g = np.zeros_like(image).astype(np.uint8)
	  b = np.zeros_like(image).astype(np.uint8)
	  
	  for l in range(0, nc):
	    idx = image == l
	    r[idx] = label_colors[l, 0]
	    g[idx] = label_colors[l, 1]
	    b[idx] = label_colors[l, 2]
	    
	  rgb = np.stack([r, g, b], axis=2)
	  return rgb
  return False

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
frame_count = 4

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

trf = T.Compose([T.Resize(360),
                 #T.CenterCrop(224),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])

video_capture = cv2.VideoCapture(0)

process_this_frame = True
rgb = None
for i in range(frame_count):
    # Grab a single frame of video
	ret, frame = video_capture.read()

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
img = Image.fromarray(rgb_frame)

inp = trf(img).unsqueeze(0)

out = dlab(inp)['out']

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
rgb = decode_segmap(om)	


#img = trf_crop(img)


person_mask = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

rgb_frame = cv2.resize(rgb_frame, (person_mask.shape[1], person_mask.shape[0]))

person_img = cv2.bitwise_and(rgb_frame, rgb_frame, mask = person_mask)


face_locations = []

height, width, channels = person_img.shape


face_locations = face_recognition.face_locations(person_img)

(_, _, bottom, _) = face_locations[0]

person_img = person_img[bottom:height, 0:width]


converted = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)
cv2.imshow('maske', skinMask)


skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(person_img, person_img, mask = skinMask)


final = cv2.subtract(person_img, skin)

gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

# Smoothing without removing edges.
gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

edges_filtered = cv2.Canny(gray_filtered, 50, 120)

cv2.imshow('Frame', edges_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

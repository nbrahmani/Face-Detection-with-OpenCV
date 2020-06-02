# Detects faces in a pic using opencv deep-learning module dnn

import numpy as np
import argparse
import cv2

# Writing arguments for command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum prob to filter weak detections")
args = vars(ap.parse_args())  # to use the argments converting


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])  #loading models
image = cv2.imread(args["image"])  #loading image
(h,w) = image.shape[:2]  # getting dimensions of image

#processing the image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))  # image, scale, size, color

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()  # compute outpuut of layer probably weights
# print(detections)
# print(detections.shape)
# print(detections[0, 0, 1, 2])
for i in range(0, detections.shape[2]):  # 3rd dimension in dimension of detections
	confidence = detections[0, 0, i, 2]

	if (confidence > args["confidence"]):
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# print(box)
		(startX, startY, endX, endY) = box.astype("int")

		text = "{:.2f}%".format(confidence*100)  #printing confidence
		y = startY - 10 if startY - 10 > 10 else startY+10  #setting pixel for putting text in proper place
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1) #image, cooeds, color, thickness
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2) #image, text, coords, font, fontsize, color, thickness

cv2.imshow("output", image)
cv2.waitKey(0)

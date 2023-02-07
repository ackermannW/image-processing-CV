# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:21:05 2022

@author: Korisnik
"""

import cv2
 
img = cv2.imread('yolo\cars.jpg')

with open('yolo\coco.names', 'r') as f:
    classes = f.read().splitlines()
 
net = cv2.dnn.readNetFromDarknet('yolo\yolov4.cfg', 'yolo\yolov4.weights')
 
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
 
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
 
for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)
 
    text = '%s: %.2f' % (classes[classIds[0]], score)
    #text ='Car'
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 255, 0), thickness=2)
 
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
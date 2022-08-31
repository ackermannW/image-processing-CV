# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 20:00:45 2022

@author: Korisnik
"""

# face detection with mtcnn on a photograph
import cv2 as cv
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
import os

# load image from file
path = os.path.join('..','..', os.getcwd(), 'images', 'faces.jpg')
image = cv.imread(path)

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
pixels = pyplot.imread(path)

# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(image)
# plot the image
pyplot.imshow(image)
# get the context for drawing boxes
ax = pyplot.gca()
# get coordinates from the first face

for  face in faces:
     dira =face['box']
     x=dira[0]
     y=dira[1]
     width=dira[2]
     height=dira[3]
     rect = Rectangle((x, y), width, height, fill=False, color='red') 
     ax.add_patch(rect)
   
x, y, width, height = faces[0]['box']
# create the shape
rect = Rectangle((x, y), width, height, fill=False, color='blue')
# draw the box
ax.add_patch(rect)
# show the plot
pyplot.show()


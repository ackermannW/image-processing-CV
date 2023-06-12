import cv2
import numpy as np
import urllib.request
import os

current_path = os.path.join(os.getcwd(), 'deep_learning', 'alex_net')
file_name = os.path.join('..', '..', os.getcwd(), 'images', 'snail.jpg')
model_path = os.path.join(current_path, 'bvlc_alexnet.caffemodel')
network_path = os.path.join(current_path, 'deploy.prototxt')

classes_file = os.path.join(current_path, 'classes.txt')

with open(classes_file, 'r') as f:
   image_net_names = f.read().split('\n')
class_names = [name.split(',')[0] for name in image_net_names]

# Download the pre-trained AlexNet model
if not os.path.exists(model_path):
    urllib.request.urlretrieve("http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel", model_path)

# Load the pre-trained AlexNet model
model = cv2.dnn.readNetFromCaffe(network_path, model_path)

# Load and preprocess the input image
image = cv2.imread(filename=file_name)
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(227, 227), mean=(104.0, 117.0, 123.0), swapRB=True)

# Set the preprocessed image as the input to the model
model.setInput(blob)

# Forward pass through the network
outputs = model.forward()

final_outputs = outputs[0]
final_outputs = final_outputs.reshape(1000, 1)
label_id = np.argmax(final_outputs)
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
final_prob = np.max(probs) * 100

out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"
cv2.putText(image, out_text, (25, 50), cv2.QT_FONT_NORMAL, 1, (0, 255, 0), 2)
cv2.imshow("Result image", image)
cv2.waitKey(0)

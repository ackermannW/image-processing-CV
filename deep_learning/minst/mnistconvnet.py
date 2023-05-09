
"""
## Setup
"""

import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""



model = keras.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu",input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add( layers.Dropout(0.2))

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add( layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add( layers.Dropout(0.2))

model.add(layers.Dense(num_classes, activation="softmax"))
model.add( layers.Dropout(0.2))

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
##  Save  model  in a file
"""

# model.save('mnist.h5')

"""
##    Model  inference (prediction)
"""
img=x_test[2345]
img=img.reshape(1,28,28,1)
score = model.predict(img)
score_classes = np.argmax(score, axis=1)
img=x_test[2345]
plt.imshow(img)
plt.waitforbuttonpress(0)
print('Cifra   je:',score_classes)




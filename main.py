import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv  
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Load and predict from an external image
for x in range(1, 5):
    img_path = r"C:\Users\kishore\Downloads\Digit-Recognition--main\Digit-Recognition--main\1.png"  # Update this with the correct path to your image file
    img = cv.imread(img_path)
    
    # Error handling: Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to load image at {img_path}. Please check the file path or file integrity.")
        continue

    # Preprocess the image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv.resize(img, (28, 28))  # Resize to match the MNIST dataset size
    img = np.invert(np.array([img]))  # Invert colors
    img = img / 255.0  # Normalize the image

    # Make predictions
    prediction = model.predict(img)
    print("----------------")
    print("The predicted value is:", np.argmax(prediction))
    print("----------------")

    # Display the image
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

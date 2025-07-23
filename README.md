# Handwritten-Digital-Recognition-with-CNN-MNIST-
A beginner-friendly deep learning repository that walks you through training and evaluating a CNN for digit recognition. Ideal for those learning computer vision and neural networks using Python and TensorFlow.

# Project Overview

- Trains a CNN on the MNIST dataset (60,000 training & 10,000 test samples)
- Achieves ~99% training accuracy and ~98.7% test accuracy
- Saves the model as `mnist_cnn_model.h5`

## Project Structure 
bash
Copy
Edit
mnist-cnn/
│
├── mnist_cnn.py           # Main Python script
├── mnist_cnn_model.h5     # Saved model (after training)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

## Code
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict and display 10 test images
predictions = model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
    plt.show()

# Model Summary

- Conv2D (32 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Flatten
- Dense (128) + ReLU
- Dense (10) + Softmax

## Author 
Vidhi Mistry
vidhimistry292@gmail.com

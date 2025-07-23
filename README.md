# Handwritten-Digital-Recognition-with-CNN-MNIST-
A beginner-friendly deep learning repository that walks you through training and evaluating a CNN for digit recognition. Ideal for those learning computer vision and neural networks using Python and TensorFlow.

# Project Overview

- Trains a CNN on the MNIST dataset (60,000 training & 10,000 test samples)
- Achieves ~99% training accuracy and ~98.7% test accuracy
- Saves the model as `mnist_cnn_model.h5`
- 
# Model Summary

- Conv2D (32 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Flatten
- Dense (128) + ReLU
- Dense (10) + Softmax

# Getting Started

# Clone the Repository
```bash
git clone https://github.com/your-username/mnist-cnn-digit-recognition.git
cd mnist-cnn-digit-recognition
```

# Install Dependencies
```bash
pip install -r requirements.txt
```

# Run the Model
```bash
python mnist_cnn.py
'''
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

## Author
Vidhi Mistry
📧 vidhimistry292@gmail.com

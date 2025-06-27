# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Upload MNIST CSV files
from google.colab import files
uploaded = files.upload()  # Select mnist_train.csv and mnist_test.csv from your computer

# Load data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Split features and labels
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = test_df.iloc[:, 0].values

# One-hot encoding
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {acc:.4f}")

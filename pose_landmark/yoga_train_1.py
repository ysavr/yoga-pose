import tensorflow as tf
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("pose_landmarks.csv")
X = data.iloc[:, :-1].values  # Features (landmarks)
y = pd.get_dummies(data.iloc[:, -1]).values  # One-hot encode labels

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=16)

# Save as TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("pose_classifier.tflite", "wb") as f:
    f.write(tflite_model)

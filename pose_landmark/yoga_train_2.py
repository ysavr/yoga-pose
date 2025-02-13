import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ 1. Load the dataset
df = pd.read_csv("pose_landmarks.csv")

# ✅ 2. Extract features (X) and labels (y)
X = df.iloc[:, :-1].values  # All columns except 'label'
y = df.iloc[:, -1].values   # Last column (label)

# ✅ 3. Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert labels to 0,1,2...

# ✅ 4. Normalize landmark values
scaler = StandardScaler()
X = scaler.fit_transform(X)

### ----- Jika menggunakan satu dataset
# ✅ 5. Split into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 6. Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")  # Output layer (num_classes)
])

# ✅ 7. Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ✅ 8. Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    verbose=1  # This shows the default progress bar with metrics
)

# Print final epoch metrics
print("\nTraining Metrics:")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")

print("\nValidation Metrics:")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# ✅ 9. Save the trained model
model.save("pose_classifier_3.h5")

# ✅ 10. Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ✅ Save TFLite model
with open("pose_classifier_3.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model training complete! Saved as pose_classifier.tflite 🎉")

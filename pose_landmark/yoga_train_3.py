import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ 1. Load the training and test datasets
train_df = pd.read_csv("yoga16_dataset_train.csv")
test_df = pd.read_csv("yoga16_dataset_test.csv")

# ✅ 2. Extract features (X) and labels (y) for training and testing
X_train = train_df.iloc[:, :-1].values  # All columns except 'label' (features)
y_train = train_df.iloc[:, -1].values  # Last column (label)

X_test = test_df.iloc[:, :-1].values   # All columns except 'label' (features)
y_test = test_df.iloc[:, -1].values    # Last column (label)

# ✅ 3. Convert labels to numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # Fit on training labels
y_test = label_encoder.transform(y_test)        # Transform test labels using the same encoder

# ✅ 4. Normalize landmark values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data
X_test = scaler.transform(X_test)        # Transform test data using the same scaler

# ✅ 5. Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer (99 features: 33 landmarks * 3 coordinates)
    tf.keras.layers.Dense(128, activation="relu"),     # Hidden layer with 128 neurons
    tf.keras.layers.Dropout(0.3),                      # Dropout for regularization
    tf.keras.layers.Dense(64, activation="relu"),      # Hidden layer with 64 neurons
    tf.keras.layers.Dropout(0.3),                      # Dropout for regularization
    tf.keras.layers.Dense(len(np.unique(y_train)), activation="softmax")  # Output layer (num_classes)
])

# ✅ 6. Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ✅ 7. Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    verbose=1  # This shows the default progress bar with metrics
)

# ✅ 8. Evaluate the model
# Print final epoch metrics
print("\nTraining Metrics:")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")

print("\nValidation Metrics:")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Set Accuracy: {test_accuracy * 100:.2f}%")

# ✅ 9. Save the trained model
model.save("yoga16_3.h5")
print("✅ Model saved as yoga16_3.h5")

# ✅ 10. Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ✅ Save TFLite model
with open("yoga16_3.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Model converted and saved as yoga16_3.tflite 🎉")
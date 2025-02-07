from tensorflow.lite.python.interpreter import Interpreter  # Correct import
import numpy as np
import json

import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load test image
img_path = os.path.join("..", "data", "images", "downdog", "00000014.jpg") # Change to your dataset path
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load TFLite model
interpreter = Interpreter(model_path="pose_classifier_2.tflite")
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label mappings
with open("pose_labels.json", "r") as f:
    labels = json.load(f)  # Example: {"0": "downdog", "1": "goddess", ...}

result = pose.process(image_rgb)
if result.pose_landmarks:
    landmarks = []
    for lm in result.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])  # Flatten (x, y, z)

    # Convert to NumPy array (1 sample, 99 features)
    pose_input = np.array([landmarks], dtype=np.float32)

# Test sample (33 landmarks * 3 = 99 values)
# test_pose_landmarks = np.random.rand(1, 99).astype(np.float32)  # Replace with real pose data

# Run inference
interpreter.set_tensor(input_details[0]["index"], pose_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])[0]

# Get predicted pose label
predicted_index = np.argmax(output_data)  # Index with highest probability
predicted_label = labels[str(predicted_index)]  # Convert index to label

print(f"Predicted Pose: {predicted_label}")
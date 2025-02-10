from tensorflow.lite.python.interpreter import Interpreter  # Correct import
import numpy as np
import json

import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load test image
img_path = os.path.join("..", "data", "images", "plank", "00000004.jpg") # Change to your dataset path
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

# Extract pose landmarks
landmarks = []
for lm in result.pose_landmarks.landmark:
    landmarks.extend([lm.x, lm.y, lm.z])  # Flatten (x, y, z)

# Convert to NumPy array (1 sample, 99 features)
pose_input = np.array([landmarks], dtype=np.float32)

# Run inference
interpreter.set_tensor(input_details[0]["index"], pose_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])[0]

# Get predicted pose label
predicted_index = np.argmax(output_data)  # Index with highest probability
predicted_label = labels[str(predicted_index)]  # Convert index to label

print(f"Predicted Pose: {predicted_label}")

# Draw pose landmarks on the image
annotated_image = image.copy()
mp_drawing.draw_landmarks(
    annotated_image,
    result.pose_landmarks,
    mp_pose.POSE_CONNECTIONS
)

# Add the predicted pose label as text on the image
cv2.putText(
    annotated_image,
    f"Predicted: {predicted_label}",
    (10, 30),  # Position of the text
    cv2.FONT_HERSHEY_SIMPLEX,
    1,  # Font scale
    (0, 255, 0),  # Green color
    2  # Thickness
)

# Display the annotated image
cv2.imshow("Pose Recognition", annotated_image)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
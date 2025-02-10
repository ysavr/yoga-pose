from tensorflow.lite.python.interpreter import Interpreter  # Correct import
import numpy as np
import json
import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load TFLite model
interpreter = Interpreter(model_path="yoga16.tflite")
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label mappings
with open("yoga16_labels.json", "r") as f:
    labels = json.load(f)  # Example: {"0": "downdog", "1": "goddess", ...}

# Define the test directory
test_dir = os.path.join("..", "data", "yoga16_dataset", "test")  # Change to your test directory path

# Initialize counters for accuracy calculation
correct_predictions = 0
total_predictions = 0

# Recursively iterate over all files in the test directory and its subdirectories
for root, dirs, files in os.walk(test_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            img_path = os.path.join(root, filename)
            print(f"Processing file: {img_path}")

            # Extract ground truth pose name from the directory name
            ground_truth_pose = os.path.basename(root)

            # Load test image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process pose landmarks using MediaPipe
            result = pose.process(image_rgb)
            if not result.pose_landmarks:
                print(f"No pose landmarks detected in: {img_path}")
                continue

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

            # Calculate prediction accuracy
            is_correct = (predicted_label == ground_truth_pose)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            # Print results
            print(f"File: {filename}")
            print(f"Ground Truth Pose: {ground_truth_pose}")
            print(f"Predicted Pose: {predicted_label}")
            print(f"Prediction Correct: {is_correct}")
            print(f"Confidence Scores: {output_data}")
            print("-" * 50)

# Calculate overall accuracy
overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
print("\nOverall Prediction Accuracy: {:.2f}%".format(overall_accuracy))
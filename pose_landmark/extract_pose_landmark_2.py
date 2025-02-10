import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define dataset folder path
dataset_folder = os.path.join("..", "data", "yoga16_dataset", "train") # Change to your dataset path

# Prepare CSV file for storing data
csv_file = "yoga16_dataset.csv"

# ✅ Correct column names for 33 landmarks (x, y, z)
columns = [f"x{i+1}" for i in range(33)] + \
          [f"y{i+1}" for i in range(33)] + \
          [f"z{i+1}" for i in range(33)] + ["label"]
df = pd.DataFrame(columns=columns)

# Loop through each class folder (e.g., "plank", "raised_hand")
for pose_label in os.listdir(dataset_folder):
    pose_path = os.path.join(dataset_folder, pose_label)
    if not os.path.isdir(pose_path):
        continue  # Skip if not a directory

    print(f"Processing {pose_label}...")

    # Loop through all images in the class folder
    for img_name in os.listdir(pose_path):
        img_path = os.path.join(pose_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue  # Skip if the image is unreadable

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Pose
        result = pose.process(rgb_image)

        if result.pose_landmarks:
            # Extract 33 landmarks (x, y, z)
            x_vals, y_vals, z_vals = [], [], []
            for lm in result.pose_landmarks.landmark:
                x_vals.append(lm.x)
                y_vals.append(lm.y)
                z_vals.append(lm.z)

            # ✅ Merge all values correctly
            landmarks = x_vals + y_vals + z_vals

            # Save to DataFrame
            new_row = pd.DataFrame([landmarks + [pose_label]], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)

# Save dataset to CSV
df.to_csv(csv_file, index=False)
print(f"✅ Dataset saved to {csv_file}")

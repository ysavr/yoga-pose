import os
import cv2
import math
import pandas as pd
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Root directory containing your images
root_dir = "data/images_processed"

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# Helper Functions
def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    """
    v1 = [p1.x - p2.x, p1.y - p2.y]
    v2 = [p3.x - p2.x, p3.y - p2.y]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    
    angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle)

def calculate_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Extract features for all images
data = []
labels = []

print(f"Root directory: {root_dir}")

# Iterate through each pose folder
for pose_name in os.listdir(root_dir):
    pose_dir = os.path.join(root_dir, pose_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(pose_dir):
        continue
    
    print(f"Processing pose: {pose_name}")
    
    # Get all image files in the pose folder
    image_files = [f for f in os.listdir(pose_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    print(f"Found {len(image_files)} images in {pose_dir}")
    
    for image_file in image_files:
        image_path = os.path.join(pose_dir, image_file)
        print(f"Processing image: {image_path}")  # Debugging line
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}. Skipping...")
            continue
        else:
            print(f"Successfully loaded image: {image_path}")
        
        # Convert the image to RGB
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error converting image to RGB: {image_path}. Error: {e}")
            continue
        
        # Process the image and detect landmarks
        results = pose.process(image_rgb)
        
        # Check if landmarks are detected
        if not results.pose_landmarks:
            print(f"No landmarks detected in image: {image_path}. Skipping...")
            continue
        else:
            print(f"Landmarks detected in image: {image_path}")
        
        # Ensure required landmarks exist
        required_indices = [11, 13, 15]  # Left shoulder, left elbow, left wrist
        if any(idx >= len(results.pose_landmarks.landmark) for idx in required_indices):
            print(f"Missing required landmarks in image: {image_path}. Skipping...")
            continue
        else:
            print(f"All required landmarks found in image: {image_path}")
        
        # Extract features
        try:
            features = [
                calculate_angle(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[15]),  # Left elbow angle
                calculate_distance(results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[11]),  # Left wrist-to-shoulder distance
                # Add more features here...
            ]
            data.append(features)
            labels.append(pose_name)  # Use the folder name as the label
            print(f"Features extracted for image: {image_path}")
        except Exception as e:
            print(f"Error extracting features for image {image_path}: {e}")

# Save the dataset
if data and labels:
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv("pose_dataset.csv", index=False)
    print("Dataset saved to pose_dataset.csv")
else:
    print("No features extracted. Please check your images and processing logic.")
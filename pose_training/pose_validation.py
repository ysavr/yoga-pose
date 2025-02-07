# detect landmark
import cv2
import mediapipe as mp
import math


# Load the trained model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib  # For loading the LabelEncoder


# # detect landmark
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# # Helper Functions
# def calculate_angle(p1, p2, p3):
#     v1 = [p1.x - p2.x, p1.y - p2.y]
#     v2 = [p3.x - p2.x, p3.y - p2.y]
#     dot_product = v1[0] * v2[0] + v1[1] * v2[1]
#     magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
#     magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
#     if magnitude_v1 == 0 or magnitude_v2 == 0:
#         return 0
#     angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
#     return math.degrees(angle)

# def calculate_distance(p1, p2):
#     return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# # Process a new image
# image_path = "data/images_processed/downdog/00000032.png"
# image = cv2.imread(image_path)
# if image is None:
#     raise ValueError(f"Failed to load image: {image_path}")

# # Convert to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Detect landmarks
# results = pose.process(image_rgb)
# if not results.pose_landmarks:
#     raise ValueError(f"No landmarks detected in image: {image_path}")

# # Extract features
# features = [
#     calculate_angle(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[15]),  # Left elbow angle
#     calculate_distance(results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[11]),  # Left wrist-to-shoulder distance
# ]
# new_features = [features]

# Load the trained TensorFlow model
model = tf.keras.models.load_model("pose_classifier.h5")

# Load the LabelEncoder used during training
label_encoder = joblib.load("label_encoder.pkl")  # Ensure this file exists

# Predict the label for new features
new_features = [[38.44742052599453,0.07852427438758505]]  # Replace with actual features
print(f"new feature: {new_features}")

predicted_probabilities = model.predict(new_features)
predicted_class = tf.argmax(predicted_probabilities, axis=1).numpy()[0]  # Get the predicted class index

# Decode the predicted class back to the original label
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

# Print the predicted pose
print(f"Predicted Pose: {predicted_label}")
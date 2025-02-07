import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ✅ 1. Load the dataset again
df = pd.read_csv("pose_landmarks.csv")

# ✅ 2. Extract labels column
y = df.iloc[:, -1].values  # Last column contains pose labels

# ✅ 3. Recreate label encoder & fit on labels
label_encoder = LabelEncoder()
label_encoder.fit(y)  # Fit on all labels

# ✅ 4. Save label mapping
with open("pose_labels.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)

print("✅ Label mapping saved to pose_labels.json 🎉")

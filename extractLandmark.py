import cv2
import os
import mediapipe as mp
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

mp_hands = mp.solutions.hands

DATASET_DIR = "Dataset"
CSV_FILE = "hand_landmarks.csv"

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2) as hands:
        results = hands.process(image_rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

        while len(landmarks) < 84:  # Ensure exactly 84 values (42 per hand)
            landmarks.append(0)

        return landmarks

data = []
all_images = []

# Collect all image paths for tqdm progress tracking
for label in sorted(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)
    if os.path.isdir(label_path):
        for image_file in os.listdir(label_path):
            all_images.append((label, os.path.join(label_path, image_file)))

# Process images with progress bar
for label, image_path in tqdm(all_images, desc="Processing Images", unit="image"):
    landmarks = extract_landmarks(image_path)
    if landmarks:
        data.append([label] + landmarks)

# Define column names
columns = ["label"] + [f"x{i}" for i in range(1, 43)] + [f"y{i}" for i in range(1, 43)]

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(CSV_FILE, index=False)

print(f"Landmark extraction complete. Data saved to {CSV_FILE}")
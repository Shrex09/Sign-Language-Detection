import cv2
import os
import numpy as np
import mediapipe as mp

GESTURES = ['hello','iloveyou','bye','stop','yes','no','peace']
DATASET_PATH = 'gesture_dataset'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_data():
    data = []
    labels = []
    for idx, gesture in enumerate(GESTURES):
        gesture_path = f"{DATASET_PATH}/{gesture}"
        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    data.append(landmarks)
                    labels.append(idx)

    data = np.array(data)
    labels = np.array(labels)
    np.save('landmarks.npy', data)
    np.save('labels.npy', labels)
    print("Preprocessing Completed!")

if __name__ == "__main__":
    preprocess_data()

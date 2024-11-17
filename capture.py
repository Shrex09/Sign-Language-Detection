import cv2
import os

GESTURES = ['hello','iloveyou','bye','stop','yes','no','peace']
DATASET_PATH = 'gesture_dataset'

os.makedirs(DATASET_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(f'{DATASET_PATH}/{gesture}', exist_ok=True)

def capture_dataset():
    cap = cv2.VideoCapture(0)
    for gesture in GESTURES:
        print(f"Capturing data for '{gesture}' gesture. Press 'q' to stop.")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Capturing {gesture}", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):  # Stop capturing for this gesture
                break
            filepath = f"{DATASET_PATH}/{gesture}/{gesture}_{count}.jpg"
            cv2.imwrite(filepath, frame)
            count += 1
    cap.release()
    cv2.destroyAllWindows()
    print("Dataset Capturing Completed!")

if __name__ == "__main__":
    capture_dataset()

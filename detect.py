import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

GESTURES = ['hello','iloveyou','bye','stop','yes','no','peace']
MODEL_PATH = 'gesture_model.h5'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def recognize_gestures():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                prediction = model.predict(np.array([landmarks]))
                gesture = GESTURES[np.argmax(prediction)]
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gestures()

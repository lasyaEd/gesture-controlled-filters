import cv2
import mediapipe as mp
import numpy as np
import torch
from train_model import GestureClassifier

# Load trained model
model = GestureClassifier(63, 6)
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Gesture to filter mapping
def apply_filter(frame, gesture):
    if gesture == 1:  # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif gesture == 2:  # Sepia
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)
    elif gesture == 3:  # Blur
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif gesture == 4:  # Edge detection
        return cv2.Canny(frame, 100, 200)
    elif gesture == 5:  # Cartoon
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_and(frame, frame, mask=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9))
    return frame  # Default: no filter

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]).flatten()
            gesture = model(torch.tensor(keypoints, dtype=torch.float32)).argmax().item()
            frame = apply_filter(frame, gesture)

    cv2.imshow("Filtered Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

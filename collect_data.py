import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define gesture labels
GESTURE_LABELS = {
    'open_palm': 0,       # No filter
    'fist': 1,            # Grayscale
    'peace_sign': 2,      # Sepia
    'thumbs_up': 3,       # Blur
    'pointing_finger': 4, # Edge detection
    'ok_sign': 5          # Cartoon
}

# Create dataset directory
DATASET_PATH = "gesture_data.pkl"

# Function to normalize keypoints
def normalize_keypoints(hand_landmarks):
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
    keypoints -= np.mean(keypoints, axis=0)  # Center the keypoints
    keypoints /= np.std(keypoints, axis=0)   # Scale the keypoints
    return keypoints.flatten()

# Data collection loop
def collect_data():
    cap = cv2.VideoCapture(0)
    collected_data = []

    for gesture, label in GESTURE_LABELS.items():
        print(f"Collecting data for {gesture}. Press 's' to start, 'q' to quit.")
        input("Press Enter to continue...")
        frame_count = 0

        while frame_count < 100:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    keypoints = normalize_keypoints(hand_landmarks)
                    collected_data.append((keypoints, label))

                    # Draw landmarks on the hand
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Gesture: {gesture} ({frame_count}/100)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                frame_count += 1
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    # Save dataset
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(collected_data, f)

    print("Data collection complete! Saved to gesture_data.pkl.")

if __name__ == "__main__":
    collect_data()
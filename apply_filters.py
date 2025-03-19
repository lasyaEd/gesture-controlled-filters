import cv2
import mediapipe as mp
import numpy as np
import torch
from train_model import GestureClassifier

class GestureFilterApp:
    def __init__(self):
        # Load trained model
        self.model = GestureClassifier(63, 6)
        self.model.load_state_dict(torch.load("gesture_model.pth"))
        self.model.eval()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Start the webcam
        self.cap = cv2.VideoCapture(0)

    def grayscale(self, image):
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def sepia(self, image):
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(image, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

    def blur(self, image, ksize=15):
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def edge_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def cartoon(self, image):
        # Apply bilateral filter to reduce noise while keeping edges sharp
        color = cv2.bilateralFilter(image, 9, 250, 250)

        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)

        # Detect edges using adaptive thresholding
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=9, C=9)

        # Convert edges to color image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Combine edge mask with the color image
        cartoon_img = cv2.bitwise_and(color, edges_colored)
        
        return cartoon_img

    def apply_filter(self, frame, gesture):
        """ Maps gesture to the corresponding filter function. Open Palm (0) applies no filter. """
        filter_functions = {
            1: self.grayscale,        # Fist -> Grayscale
            2: self.sepia,            # Peace sign -> Sepia
            3: self.blur,             # Thumbs up -> Blur
            4: self.edge_detection,   # Pointing index finger -> Edge detection
            5: self.cartoon           # OK sign -> Cartoon
        }
        return filter_functions.get(gesture, lambda x: x)(frame)  # Default: No filter

    def normalize_keypoints(self, hand_landmarks):
        """ Normalizes hand keypoints for better prediction accuracy. """
        keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

        # Normalize keypoints relative to hand center
        keypoints -= np.mean(keypoints, axis=0)  # Center the keypoints
        keypoints /= np.std(keypoints, axis=0)   # Scale the keypoints

        return keypoints.flatten()

    def run(self):
        """ Real-time gesture detection and filter application """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(frame_rgb)
            gesture = 0  # Default: Open Palm (No Filter)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract and normalize keypoints
                    keypoints = self.normalize_keypoints(hand_landmarks)

                    try:
                        # Convert to PyTorch tensor and predict gesture
                        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            gesture = self.model(keypoints_tensor).argmax().item()
                        
                        print(f"Detected gesture: {gesture}")

                    except Exception as e:
                        print(f"Error in gesture classification: {e}")
                        gesture = 0  # Default to Open Palm (No filter)

                    # Draw landmarks on the hand
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Apply the corresponding filter
            filtered_frame = self.apply_filter(frame.copy(), gesture)

            # Display original and filtered video
            combined = np.hstack((frame, filtered_frame))
            cv2.imshow("Gesture-Controlled Filters", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureFilterApp()
    app.run()

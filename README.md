# 🎭 Gesture-Controlled Image Filters using OpenCV & MediaPipe 🎨

## 📌 Overview
This project implements **real-time hand gesture recognition** to apply different image filters using a webcam.  
The system detects **six different hand gestures** and applies corresponding filters to the webcam feed.

### **Supported Gestures & Filters:**
| Gesture | Effect |
|---------|--------|
| ✋ **Open Palm** | No filter (Default) |
| ✊ **Fist** | Grayscale |
| ✌️ **Peace Sign** | Sepia |
| 👍 **Thumbs Up** | Blur |
| ☝️ **Pointing Finger** | Edge Detection |
| 👌 **OK Sign** | Cartoon Effect |

---

## 🔧 **Installation**
Ensure you have Python installed (>=3.7). Then, install the required dependencies:

```bash
pip install opencv-python mediapipe numpy torch torchvision scikit-learn pickle-mixin
```

Alternatively, install from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🚀 **Project Structure**
```plaintext
gesture-controlled-filters/
│— collect_data.py       # Collects hand keypoints for model training
│— train_model.py        # Trains a gesture classification model
│— apply_filters.py      # Runs real-time gesture detection & filtering
│— gesture_data.pkl      # Saved dataset of hand keypoints (generated by collect_data.py)
│— gesture_model.pth     # Trained PyTorch model (generated by train_model.py)
│— requirements.txt      # Required dependencies
│— README.md             # Project documentation
```

---

## 🎥 **How to Use**
### 1️⃣ **Collect Gesture Data**
Run this script to collect **200 frames per gesture** using MediaPipe Hands:

```bash
python collect_data.py
```
🛠 **Instructions:**
- Press **Enter** to start collecting for each gesture.
- Hold the gesture steady and slightly vary angles.
- Captures **200 samples per gesture**.
- Press **'q'** to quit data collection.

### 2️⃣ **Train the Gesture Classification Model**
Once data is collected, train the **MLP neural network**:

```bash
python train_model.py
```
💪 This will generate `gesture_model.pth`, which is the trained model.

### 3️⃣ **Run Real-Time Gesture Recognition & Filtering**
```bash
python apply_filters.py
```
👀 The webcam will display **both the original frame and the filtered frame**.  
Try **different gestures** to see filters change in real time!

---

## 🛠 **Troubleshooting**
### ❌ **No webcam feed appears**
- Ensure your **camera is enabled** in system settings.
- Try running with `cv2.VideoCapture(1)` if using an external webcam.

### ❌ **No gestures detected**
- Increase `min_detection_confidence` in `collect_data.py`:
  ```python
  hands = mp.solutions.hands.Hands(min_detection_confidence=0.3)
  ```
- Ensure **good lighting** and **clear background**.

### ❌ **Filters not applying correctly**
- Print model predictions:
  ```python
  print(f"Detected gesture: {gesture}")
  ```
- If misclassification occurs, **collect more data** and **retrain the model**.

### ❌ **Cartoon filter is not showing correctly**
- Adjust `cv2.adaptiveThreshold()` in `cartoon()`:
  ```python
  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, blockSize=9, C=9)
  ```

---

## 🔥 **Upcoming Improvements**
🔹 Add **multiple hand support**  
🔹 Improve **gesture accuracy** with more training data  
🔹 Convert project into a **Streamlit app for a GUI**  

---

## 👨‍💻 **Author**
Developed by **[Your Name]**  
GitHub: [Your GitHub Profile](https://github.com/YourGitHubUsername)  

---

## ⭐ **Contributing**
Feel free to **fork this repo** and submit pull requests! Suggestions and improvements are always welcome.  

🚀 **Happy Coding!** 🎬✨


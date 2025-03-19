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


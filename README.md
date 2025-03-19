# gesture-controlled-filters
A gesture-controlled image filtering system using MediaPipe Hands and a simple ML model for classification.

## How This Works
1. Gesture Detection:

Uses MediaPipe Hands to extract 21 keypoints.
Normalizes keypoints and feeds them to the trained neural network.
The model predicts the gesture class (0-5).

2. Filter Application:

Uses a dictionary (filter_functions) to map gesture labels to corresponding filter functions.
Applies the selected filter to the frame.

3. Real-time Video Processing:

Displays original + filtered frames side by side.
Runs until 'q' is pressed.


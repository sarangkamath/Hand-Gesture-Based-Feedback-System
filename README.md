# Hand-Gesture-Based-Feedback-System
### Overview
The Hand Gesture-Based Feedback System is an innovative application that leverages computer vision and machine learning to recognize and classify hand gestures in real-time. It provides a contactless, intuitive, and hygienic way to collect feedback, making it suitable for events, exhibitions, kiosks, and other public settings. The system processes gestures as positive or negative votes, logs them, and provides real-time visual analytics using an interactive dashboard.

### Features
- Real-Time Gesture Recognition: Leverages MediaPipe for robust hand tracking and keypoint extraction.
- Machine Learning Model: A TensorFlow-based neural network trained to classify gestures.
- Interactive Dashboard: Built using Streamlit,which features live video field of gesture detection and toast notifications confirming feedback
- Feedback Logging: Stores user responses in a CSV file for further analysis.
- Customizable UI: Enhanced with CSS for modern, user-friendly notifications.

### Use Cases
1. Events and Exhibitions: Collect attendee feedback through simple gestures.
2. Interactive Kiosks: Enable hands-free feedback in public spaces.
3. Education: Gauge understanding or engagement in classrooms and workshops.
4. Retail and Hospitality: Gather customer opinions hygienically.
5. Healthcare: Ensure safe and touchless feedback collection.

### Technologies Used
- Python: Core programming language for the system.
- Streamlit: For building the interactive web-based dashboard.
- OpenCV: For handling the video feed and gesture recognition.
- MediaPipe: For detecting and tracking hand keypoints.
- TensorFlow: For training and deploying the neural network model.
- Pandas: For managing and logging feedback data.
- CSS: For styling toast notifications and enhancing the UI.

## Running the application
1. To launch the application, run the following command in your terminal:
```bash
streamlit run main.py
```
2. A web-based interface will open in your default browser
3. Interact:
   - Perform gestures in front of the webcam
   - The system will classify the gestures and log them as positive or negative votes
   - Notifications will confirm your input and log them in the CSV file in real time

### Gesture Classes
| **Gesture**       | **Classification** | **Purpose**                    |
|--------------------|--------------------|---------------------------------|
| 👍 Thumbs Up       | Positive           | Approval or liking.            |
| 👎 Thumbs Down     | Negative           | Didn't like it or did not meet expectations.       |
| 🫶 Heart Gesture   | Positive           | Loved it!          |
| 👌 OK Sign         | Positive           | Everything is fine or perfect. |

## Acknowledgements
- MediaPipe: For its efficient hand tracking pipeline
- Streamlit: For simplifying web-based application development




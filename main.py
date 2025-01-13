import csv
import argparse
import time
import os
import datetime as dt
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp
import numpy as np
import streamlit as st

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

import streamlit as st
from streamlit_option_menu import option_menu

log_file= "gesture_logs.csv"
last_action_time= 0
cooldown_period= 5
positive_vote= 0
negative_vote= 0

if "last_vote_time" not in st.session_state:
    st.session_state["last_vote_time"] = 0
if "notification" not in st.session_state:
    st.session_state["notification"] = ""

def get_args():
    parser= argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="Camera Device Index")
    parser.add_argument("--width", type=int, default=960, help="Capture Width")
    parser.add_argument("--height", type=int, default=540, help="Capture Height")

    parser.add_argument("--use_static_image_mode", action="store_true", help="Enable static image mode")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7, help="Minimum detection confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="Minimum tracking confidence")

    return parser.parse_args()

def load_labels():
    def read_csv(file_path):
        with open(file_path, encoding="utf-8-sig") as f:
            return [row[0] for row in csv.reader(f)]
    
    keypoint_labels= read_csv('model/keypoint_classifier/keypoint_classifier_label.csv')
    point_history_labels= read_csv('model/point_history_classifier/point_history_classifier_label.csv')

    return keypoint_labels, point_history_labels

def camera_setup(cap_device, cap_width, cap_height):
    cap= cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    return cap

def initialize_models(use_static_image_mode, min_detection_confidence, min_tracking_confidence):
    #Custom classifiers
    keypoint_classifier= KeyPointClassifier()
    point_history_classifier= PointHistoryClassifier()

    #Mediapipe Hands Model
    mp_hands= mp.solutions.hands
    hands= mp_hands.Hands(
        static_image_mode= use_static_image_mode,
        max_num_hands= 2,
        min_detection_confidence= min_detection_confidence,
        min_tracking_confidence= min_tracking_confidence,
    )

    return hands, keypoint_classifier, point_history_classifier

def select_mode(key, mode):
    number= -1
    if 48 <= key <=57:   #0 ~ 9
        number= key-48
    if key == 110:   #n
        mode= 0
    if key == 107:  #k
        mode= 1
    if key == 104:  #h
        mode= 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height= image.shape[1], image.shape[0]

    #Extract landmark coordinates as numpy array
    landmarks_array= np.array([
        [min(int(landmark.x * image_width), image_width-1),
         min(int(landmark.y * image_height), image_height-1)]
         for landmark in landmarks.landmark
    ])

    #using opencv to calculate the bouding rectangle
    x, y, w, h= cv.boundingRect(landmarks_array)
    return [x, y, x+w, y+h]

def calc_landmark_list(image, landmarks):
    image_width, image_height= image.shape[1], image.shape[0]
   
    #Keypoint
    landmark_point= np.array([
        [min(int(landmark.x * image_width), image_width-1),
         min(int(landmark.y * image_height), image_height-1)]
         for landmark in landmarks.landmark
    ])

    return landmark_point

def pre_process_landmark(landmark_list):
    #convert to numpy array
    temp_landmark_array= np.array(landmark_list, dtype=np.float32)

    #convert to relative coordinates
    base_x, base_y= temp_landmark_array[0]
    temp_landmark_array[:, 0]-= base_x
    temp_landmark_array[:, 1]-= base_y

    #convert to 1 dimensional list and normalize
    temp_landmark_list= temp_landmark_array.flatten()
    max_value= np.max(np.abs(temp_landmark_list))
    temp_landmark_list/= max_value

    return temp_landmark_list.tolist()

def pre_process_point_history(image, point_history):
    image_width, image_height= image.shape[1], image.shape[0]
    temp_point_history= np.array(point_history, dtype=np.float32)

    if len(temp_point_history)>0:
        base_x, base_y= temp_point_history[0]
        temp_point_history[:, 0]= (temp_point_history[:, 0]- base_x)/ image_width
        temp_point_history[:, 1]= (temp_point_history[:, 1]- base_y)/ image_height

    temp_point_history= temp_point_history.flatten()

    return temp_point_history.tolist()

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path= 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer= csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path= 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer= csv.writer(f)
            writer.writerow([number, *point_history_list])
    return 

def draw_landmarks(image, landmark_point):
    if len(landmark_point>0):
        #thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)  
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        #index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        #middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        #ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        #little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        #palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

        #Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0: #wrist 1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1: #wrist 2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2: #thumb base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3: #thumb 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4: #thumb fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5: #index finger base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6: #index finger 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7: #index finger 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8: #index finger fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9: #middle finger base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10: #middle finger 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11: #middle finger 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12: #middle finger fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13: #ring finger base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14: #ring finger 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15: #ring finger 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16: #ring finger fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17: #little finger base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18: #little finger 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19: #little finger 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20: #little finger fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        #outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1]-22), (0, 0, 0), -1)

    info_text= handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text= info_text + ":" + hand_sign_text
    cv.putText(image, info_text, (brect[0]+5, brect[1]-4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0]!= 0 and point[1]!= 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string= ['Logging  Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "Mode:" + mode_string[mode-1], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "Num:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image

def perform_action():
    global last_action_time
    current_time= time.time()

    if current_time - last_action_time > cooldown_period:
        last_action_time= current_time
        return True
    else:
        return False
    
def log_gesture(response, gesture_type):
    timestamp= dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    check_file= os.path.isfile(log_file)

    with open(log_file, 'a', newline="") as file:
        writer= csv.writer(file)
        if not check_file:
            writer.writerow(["Timestamp", "Response", "Gesture Type"]) 
        writer.writerow([timestamp, response, gesture_type])

def show_toast(message, bg_colour, duration=3):
    toast_placeholder = st.empty()
    toast_placeholder.markdown(
        f"""
        <div style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            background-color: {bg_colour};
            color: white;
            border-radius: 5px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Schedule the notification to disappear after `duration` seconds
    st.session_state["notification"] = {"placeholder": toast_placeholder, "end_time": time.time() + duration}

#---STREAMLIT CODE---
with st.sidebar:
    selected = option_menu(
        "Control Panel",
        ["Home", "Overview", "About", "Settings"],
        icons=["house", "layout-text-sidebar", "info-circle", "gear"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Overview":
    st.title("Overview")
    st.markdown("""
    <p style="text-align: justify;">
    This project, titled Hand Gesture-Based Feedback System, aims to provide a seamless and innovative method for capturing real-time feedback using hand gestures. Leveraging computer vision techniques with OpenCV and MediaPipe, the system identifies and classifies hand gestures into positive and negative responses. The primary objective is to simplify feedback collection in events, exhibitions, or interactive setups while ensuring accuracy and efficiency.
    Key functionalities include real-time gesture recognition, dynamic logging of votes into a CSV file, and a user-friendly interface developed using Streamlit. Interactive toast notifications provide immediate acknowledgment of user responses, enhancing user engagement. The system also incorporates a visual analytics section with bar chart representation of positive and negative votes for easy interpretation of feedback trends.
    By integrating OpenCV for image processing, MediaPipe for robust hand tracking, and Streamlit for an interactive web interface, the project demonstrates a practical application of AI and computer vision technologies in the domain of feedback systems.</p>""",
    unsafe_allow_html=True)

    
elif selected == "About":
    st.title("About")
    st.markdown("""
    <p style="text-align: justify;">
    The Hand Gesture-Based Feedback System is designed to provide a seamless, contactless solution for collecting real-time feedback using hand gestures. 
    This innovative system leverages computer vision and machine learning to interpret user gestures, classifying them as positive or negative responses.
    It aims to enhance engagement, streamline feedback collection, and eliminate the need for physical touch, making it ideal for public and interactive environments.</p>""",
    unsafe_allow_html=True)
    st.subheader("Purpose")
    st.markdown("""
    - To enable efficient feedback collection in scenarios where traditional methods may not be practical.
    - To demonstrate the potential of AI and gesture recognition in improving user interaction and experience.
    - To provide a real-time, visual representation of audience feedback for quick decision-making and analysis.""", unsafe_allow_html=True)
    st.subheader("Use Cases")
    st.markdown("""
    - Events and Exhibitions: Collect feedback from attendees through simple hand gestures, ensuring quick and intuitive interactions.
    - Kiosks and Public Displays: Deploy at information booths or interactive displays for user feedback without requiring physical input devices.
    - Education and Training: Use in classrooms or workshops for gauging participant understanding and engagement.
    - Retail and Hospitality: Gather customer opinions on services or products in a contactless manner.
    - Healthcare: Implement in clinics or hospitals for feedback collection while maintaining hygiene standards.""", unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: justify;">
    This project demonstrates how technology can make feedback systems more accessible, user-friendly, and adaptable to various industries, showcasing the practical applications of AI and computer vision in real-world scenarios.</p>""", unsafe_allow_html=True)


elif selected == "Settings":
    st.title("Settings")
    min_detection_confidence = st.slider("Min Detection Confidence", 0.0, 1.0, 0.7)
    min_tracking_confidence = st.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)


elif selected == "Home":
    st.title("Hand Gesture-Based Feedback System")

    #video container
    video_container= st.image([])

    #initializing the models
    args= get_args()

    hands, keypoint_classifier, point_history_classifier= initialize_models(args.use_static_image_mode,
                                                                            args.min_detection_confidence,
                                                                            args.min_tracking_confidence)
    keypoint_labels, point_history_labels= load_labels()
    #FPS calculation and coordinate history
    cvFpsCalc= CvFpsCalc(buffer_len=10)
    history_length= 16
    point_history= deque(maxlen=history_length)
    finger_gesture_history= deque(maxlen=history_length)

    use_brect= True
    mode=0

    #video capture
    cap= cv.VideoCapture(0)

    while cap.isOpened():
        fps= cvFpsCalc.get()

        #Process key (esc: end)
        key= cv.waitKey(10)
        if key == 27: #esc
            break
        number, mode= select_mode(key, mode)

        ret, image= cap.read()
        if not ret or image is None:
            st.warning("Unable to access webcam!")
            break

        #image processing
        image= cv.flip(image, 1) #mirror image
        debug_image= image.copy()

        #detection implementation
        image= cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable= False
        results= hands.process(image)
        image.flags.writeable= True

        if st.session_state["notification"]:
            notification = st.session_state["notification"]
            if time.time() > notification["end_time"]:
                notification["placeholder"].empty()
                st.session_state["notification"] = ""

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                #bounding box calculation
                brect= calc_bounding_rect(debug_image, hand_landmarks)

                #landmark calculation
                landmark_list= calc_landmark_list(debug_image, hand_landmarks)

                #conversion to relative coordinates/ normalized coordinates
                pre_processed_landmark_list= pre_process_landmark(landmark_list)
                pre_processed_point_history_list= pre_process_point_history(debug_image, point_history)
                #write to dataset
                logging_csv= (number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                #hand sign classification
                hand_sign_id= keypoint_classifier(pre_processed_landmark_list)
                check_cooldown= perform_action()
                notification_placeholder= st.empty()

                if check_cooldown:
                    if hand_sign_id == 0:
                        positive_vote += 1
                        log_gesture("Like", "Postive Response")
                        show_toast("Glad to know that you enjoyed the experience!", "#4CAF50")
                        
                    elif hand_sign_id == 1:
                        negative_vote += 1
                        log_gesture("Dislike","Negative Response")  
                        show_toast("We're sorry we couldn't meet your expectations!", "#E74C3C")

                    elif hand_sign_id == 2:
                        positive_vote +=1
                        log_gesture("Good", "Positive Response")
                        show_toast("Glad to know that you enjoyed the experience!", "#4CAF50")
                    
                    elif hand_sign_id == 6:
                        positive_vote +=1
                        log_gesture("Loved it!", "Positive Response")
                        show_toast("Glad to know that you enjoyed the experience!", "#4CAF50")
                        
                #finger gesture classification
                finger_gesture_id= 0
                point_history_len= len(pre_processed_point_history_list)
                
                if point_history_len == (history_length*2):
                    finger_gesture_id= point_history_classifier(pre_processed_point_history_list)
                
                #calculate the gesture IDs in latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id= Counter(finger_gesture_history).most_common()

                #drawing
                debug_image= draw_bounding_rect(use_brect, debug_image, brect)
                debug_image= draw_landmarks(debug_image, landmark_list)
                debug_image= draw_info_text(debug_image, brect, handedness, keypoint_labels[hand_sign_id])
        
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, 0, 0)

        debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        video_container.image(debug_image)

    cap.release()
    hands.close()
    




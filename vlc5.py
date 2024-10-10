import cv2
import mediapipe as mp
import pyautogui
import time
import configparser

# Load configuration settings
config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters
gesture_threshold = config.getfloat('Gestures', 'threshold')
cooldown_time = config.getfloat('Gestures', 'cooldown')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
webcam = cv2.VideoCapture(0)

# Gesture control mapping
def control_vlc(action):
    if action == "volume_up":
        pyautogui.press('volumeup')
        print("Volume increased")
    elif action == "volume_down":
        pyautogui.press('volumedown')
        print("Volume decreased")
    elif action == "play_pause":
        pyautogui.press('space')
        print("Play/Pause toggled")
    elif action == "right":
        pyautogui.hotkey('ctrl', 'right')
        print("Seek forward")
    elif action == "left":
        pyautogui.hotkey('ctrl', 'left')
        print("Seek backward")

# Initialize the last action time
last_action_time = time.time()

def calculate_distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    little_tip = landmarks[20]

    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    thumb_middle_distance = calculate_distance(thumb_tip, middle_tip)
    thumb_little_distance = calculate_distance(thumb_tip, little_tip)

    if thumb_index_distance > gesture_threshold:
        return "volume_up"
    elif thumb_index_distance < gesture_threshold and thumb_index_distance > gesture_threshold / 2:
        return "volume_down"
    elif thumb_middle_distance > gesture_threshold:
        return "right"
    elif thumb_little_distance > gesture_threshold:
        return "left"
    else:
        return "play_pause"

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = recognize_gesture(hand_landmarks.landmark)
            if current_time - last_action_time > cooldown_time:
                control_vlc(gesture)
                last_action_time = current_time

    cv2.imshow("Gesture-based VLC Control", frame)

    if cv2.waitKey(50) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()

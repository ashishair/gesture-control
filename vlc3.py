import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
webcam = cv2.VideoCapture(0)

# Gesture control mapping
def control_vlc(volume_change=None, action=None):
    if volume_change:
        pyautogui.press(volume_change)
        print(f"Volume changed: {volume_change}")
    
    elif action == "space": 
        pyautogui.press("space")
        print("Play/Pause toggled")
    
    elif action:
        pyautogui.hotkey('ctrl', action)
        print(f"Action: {action}")

# Initialize the last action time
last_action_time = time.time()

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    current_time = time.time()
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            y_wrist = int(landmarks[0].y * frame_height)
            x_thumb = int(landmarks[4].x * frame_width)
            y_thumb = int(landmarks[4].y * frame_height)
            x_index = int(landmarks[8].x * frame_width)
            y_index = int(landmarks[8].y * frame_height)
            x_middle = int(landmarks[12].x * frame_width)
            y_middle = int(landmarks[12].y * frame_height)
            y_middlepip = int(landmarks[9].y * frame_height)
            x_ring = int(landmarks[16].x * frame_width)
            y_ring = int(landmarks[16].y * frame_height)
            x_little = int(landmarks[20].x * frame_width)
            y_little = int(landmarks[20].y * frame_height)
            
            # Draw circles at thumb and index fingertips
            cv2.circle(frame, (x_thumb, y_thumb), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
            
            dis_middle_thumb = (x_middle -  x_thumb)
            dis_little_thumb = (x_little - x_thumb)
            
            # Calculate distance between thumb and index finger
            distance = ((x_thumb - x_index)**2 + (y_thumb - y_index)**2) ** 0.5
            print(f"Distance: {distance}")

            
            
            
            # Control VLC based on distance and time
            if current_time - last_action_time > 0.2:
                if distance > 60:
                    control_vlc(volume_change='volumeup')
                    last_action_time = current_time
                elif 30 < distance < 60:
                    control_vlc(volume_change='volumedown')
                    last_action_time = current_time
                
                    
                elif dis_middle_thumb > 70:
                    control_vlc(action="right")
                    last_action_time = current_time
                elif dis_little_thumb > 70:
                    control_vlc(action="left")
                    last_action_time = current_time
                
    
                
            
           
           
               
    
    cv2.imshow("Gesture-based VLC Control", frame)
    
    # Exit on pressing the Esc key
    if cv2.waitKey(50) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()

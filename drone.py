import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
webcam = cv2.VideoCapture(0)

# Gesture control mapping
def control_drone(command):
    if command == "up":
        print("Drone moving up")
        # Add code to send 'up' command to the drone
    elif command == "down":
        print("Drone moving down")
        # Add code to send 'down' command to the drone
    elif command == "left":
        print("Drone moving left")
        # Add code to send 'left' command to the drone
    elif command == "right":
        print("Drone moving right")
        # Add code to send 'right' command to the drone
    elif command == "forward":
        print("Drone moving forward")
        # Add code to send 'forward' command to the drone
    elif command == "backward":
        print("Drone moving backward")
        # Add code to send 'backward' command to the drone

# Initialize the last action time
last_action_time = time.time()

# Gesture recognition thresholds
gesture_threshold = 70  # Adjust based on testing

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
            
            x_thumb = int(landmarks[4].x * frame_width)
            y_thumb = int(landmarks[4].y * frame_height)
            x_index = int(landmarks[8].x * frame_width)
            y_index = int(landmarks[8].y * frame_height)
            x_middle = int(landmarks[12].x * frame_width)
            y_middle = int(landmarks[12].y * frame_height)
            x_little = int(landmarks[20].x * frame_width)
            y_little = int(landmarks[20].y * frame_height)
            
            # Draw circles at thumb and index fingertips
            cv2.circle(frame, (x_thumb, y_thumb), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
            
            # Calculate distances and directions
            distance_index_thumb = ((x_thumb - x_index)**2 + (y_thumb - y_index)**2) ** 0.5
            distance_middle_thumb = ((x_thumb - x_middle)**2 + (y_thumb - y_middle)**2) ** 0.5
            distance_little_thumb = ((x_thumb - x_little)**2 + (y_thumb - y_little)**2) ** 0.5
            hand_center_x = (x_thumb + x_index) / 2
            hand_center_y = (y_thumb + y_index) / 2
            
            # Control Drone based on distance and time
            if current_time - last_action_time > 0.2:
                if distance_index_thumb > gesture_threshold:
                    control_drone("up")
                    last_action_time = current_time
                elif distance_index_thumb < gesture_threshold and distance_index_thumb > gesture_threshold / 2:
                    control_drone("down")
                    last_action_time = current_time
                elif distance_middle_thumb > gesture_threshold:
                    control_drone("forward")
                    last_action_time = current_time
                elif distance_little_thumb > gesture_threshold:
                    control_drone("backward")
                    last_action_time = current_time
                elif hand_center_x < frame_width / 2 - 50:
                    control_drone("left")
                    last_action_time = current_time
                elif hand_center_x > frame_width / 2 + 50:
                    control_drone("right")
                    last_action_time = current_time
    
    cv2.imshow("Gesture-based Drone Control", frame)
    
    # Exit on pressing the Esc key
    if cv2.waitKey(50) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()

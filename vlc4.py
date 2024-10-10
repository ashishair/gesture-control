import cv2
import math
import pyautogui
import time

class Hand:

    def __init__(self, binary, masked, raw, frame):
        self.masked = masked
        self.binary = binary
        self._raw = raw
        self.frame = frame
        self.contours = []
        self.outline = self.draw_outline()
        self.fingertips = self.extract_fingertips()

    def draw_outline(self, min_area=10000, color=(0, 255, 0), thickness=2):
        contours, _ = cv2.findContours(
            self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        palm_area = 0
        flag = None
        cnt = None
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > palm_area:
                palm_area = area
                flag = i
        if flag is not None and palm_area > min_area:
            cnt = contours[flag]
            self.contours = cnt
            cpy = self.frame.copy()
            cv2.drawContours(cpy, [cnt], 0, color, thickness)
            return cpy
        else:
            return self.frame

    def extract_fingertips(self, filter_value=50):
        cnt = self.contours
        if len(cnt) == 0:
            return cnt
        points = []
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            end = tuple(cnt[e][0])
            points.append(end)
        filtered = self.filter_points(points, filter_value)

        filtered.sort(key=lambda point: point[1])
        return [pt for idx, pt in zip(range(5), filtered)]

    def filter_points(self, points, filter_value):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i] and points[j] and self.dist(points[i], points[j]) < filter_value:
                    points[j] = None
        filtered = []
        for point in points:
            if point is not None:
                filtered.append(point)
        return filtered

    def get_center_of_mass(self):
        if len(self.contours) == 0:
            return None
        M = cv2.moments(self.contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def dist(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)

def detect_face(frame, block=False, colour=(0, 0, 0)):
    fill = [1, -1][block]
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    area = 0
    X = Y = W = H = 0
    for (x, y, w, h) in faces:
        if w * h > area:
            area = w * h
            X, Y, W, H = x, y, w, h
    cv2.rectangle(frame, (X, Y), (X + W, Y + H), colour, fill)

def capture_histogram(source=0):
    cap = cv2.VideoCapture(source)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1000, 600))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Place region of the hand inside box and press `A`",
                    (5, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (500, 100), (580, 180), (105, 105, 105), 2)
        box = frame[105:175, 505:575]

        cv2.imshow("Capture Histogram", frame)
        key = cv2.waitKey(10)
        if key == 97:
            object_color = box
            cv2.destroyAllWindows()
            break
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
    object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
                               [12, 15], [0, 180, 0, 256])

    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    cap.release()
    return object_hist

def locate_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject(
        [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cv2.filter2D(object_segment, -1, disc, object_segment)

    _, segment_thresh = cv2.threshold(
        object_segment, 70, 255, cv2.THRESH_BINARY)

    # apply some image operations to enhance image
    kernel = None
    eroded = cv2.erode(segment_thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # masking
    masked = cv2.bitwise_and(frame, frame, mask=closing)

    return closing, masked, segment_thresh

def detect_hand(frame, hist):
    detected_hand, masked, raw = locate_object(frame, hist)
    return Hand(detected_hand, masked, raw, frame)

def control_vlc(action):
    if action == "play_pause":
        pyautogui.press("space")
        print("Play/Pause toggled")
    elif action == "volume_up":
        pyautogui.press("volumeup")
        print("Volume Up")
    elif action == "volume_down":
        pyautogui.press("volumedown")
        print("Volume Down")
    elif action == "skip_forward":
        pyautogui.hotkey("ctrl", "right")
        print("Skip Forward")
    elif action == "skip_backward":
        pyautogui.hotkey("ctrl", "left")
        print("Skip Backward")

cap = cv2.VideoCapture(0)
hist = capture_histogram(source=0)
last_action_time = time.time()

while True:
    ret, frame = cap.read()
    
    # detect the hand
    hand = detect_hand(frame, hist)
    
    # plot the fingertips
    for fingertip in hand.fingertips:
        cv2.circle(hand.outline, fingertip, 5, (0, 0, 255), -1)
    
    # Determine the action based on the number of fingertips
    num_fingertips = len(hand.fingertips)
    current_time = time.time()

    if current_time - last_action_time > 0.2:
        if num_fingertips == 1:
            control_vlc("play_pause")
            last_action_time = current_time
        elif num_fingertips == 2:
            control_vlc("volume_up")
            last_action_time = current_time
        elif num_fingertips == 3:
            control_vlc("volume_down")
            last_action_time = current_time
        elif num_fingertips == 4:
            control_vlc("skip_forward")
            last_action_time = current_time
        elif num_fingertips == 5:
            control_vlc("skip_backward")
            last_action_time = current_time

    cv2.imshow("Handy", hand.outline)
    
    k = cv2.waitKey(5)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

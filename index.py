import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

MIN_DISTANCE = 20
MAX_DISTANCE = 200

def map_distance_to_brightness(distance, min_d, max_d):
    distance = max(min_d, min(max_d, distance))
    brightness = np.interp(distance, [min_d, max_d], [1, 100])
    return int(brightness)

prev_brightness = -1  # To avoid setting brightness too frequently

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))

        cv2.circle(frame, thumb_pos, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, index_pos, 10, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, thumb_pos, index_pos, (255, 0, 255), 3)

        distance = calculate_distance(thumb_pos, index_pos)
        brightness = map_distance_to_brightness(distance, MIN_DISTANCE, MAX_DISTANCE)

        # Set screen brightness only if value changed significantly to avoid overload
        if abs(brightness - prev_brightness) >= 2:
            sbc.set_brightness(brightness)
            prev_brightness = brightness

        cv2.putText(frame, f'Brightness: {brightness}%', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

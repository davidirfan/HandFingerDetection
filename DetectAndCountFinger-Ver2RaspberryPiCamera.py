# this program using Raspberry Pi camera

from picamera2 import Picamera2
import cv2
import time
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)  # Adjusted parameters
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Tips of fingers
    thumb_tip = 4  # Thumb tip
    fingers = []

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Finger is raised
        else:
            fingers.append(0)  # Finger is not raised

    # Check thumb position
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        fingers.append(1)  # Thumb is raised
    else:
        fingers.append(0)  # Thumb is not raised

    return fingers

# Initialize the camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (848, 480), "format": "BGR888"})
picam2.configure(preview_config)
# Enable autofocus
picam2.set_controls({"AfMode": 2})  # Continuous autofocus mode
picam2.start()
time.sleep(1)

try:
    while True:
        # Capture image from Picamera2
        img = picam2.capture_array()

        # Rotate the image by 180 degrees
        # img = cv2.rotate(img, cv2.ROTATE_180)

        # Preprocessing: Convert to grayscale, apply Gaussian blur, then convert back to BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image
        img_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for MediaPipe

        # Process the image with MediaPipe
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count fingers
                fingers_status = count_fingers(hand_landmarks)
                num_fingers = fingers_status.count(1)

                # Display results
                cv2.putText(img, f'Fingers: {num_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # No hand detected
            cv2.putText(img, 'No hand detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the camera feed
        cv2.imshow('Hand Tracking', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.close()
    cv2.destroyAllWindows()


# finger detection and counter
# this program using local camera like from laptop

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB as required by MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark

            # Finger counting logic
            fingers = []
            for i, lm in enumerate(landmarks):
                # Thumb (index 4)
                if i == 4:
                    if lm.x < landmarks[3].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Other fingers (index 8, 12, 16, 20)
                if i in [8, 12, 16, 20]:
                    if lm.y < landmarks[i - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            finger_count = fingers.count(1)
            # Display finger count on the screen
            cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # No hand detected
        cv2.putText(frame, 'No finger detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with finger count
    cv2.imshow('Finger Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

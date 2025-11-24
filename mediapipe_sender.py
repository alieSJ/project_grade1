import cv2
import mediapipe as mp
import zmq
import json

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ZeroMQ PUSH socket
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5555")   # C++ 从这里接收

cap = cv2.VideoCapture(0)

print("Python MediaPipe sender started...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)

    landmarks63 = []

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        for p in lm.landmark:
            landmarks63 += [p.x, p.y, p.z]

        # JSON 打包后发送
        socket.send_json(landmarks63)
    else:
        socket.send_json([])

    cv2.imshow("Python MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()


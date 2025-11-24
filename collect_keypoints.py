import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands

# 存放所有样本
X_data = []
y_data = []

# 手势类别映射
GESTURE_LABEL = {
    "wave": 0,
    "fist": 1,
    "open_palm": 2
}

gesture_name = input("请输入当前采集的手势名称 (wave / fist / open_palm): ")
label = GESTURE_LABEL[gesture_name]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    print("开始采集关键点数据，按 q 保存退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            # 提取 21 个关键点
            hand = result.multi_hand_landmarks[0]
            keypoints = []

            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            if len(keypoints) == 63:
                X_data.append(keypoints)
                y_data.append(label)

        # 显示画面
        cv2.putText(frame, f"Collecting: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Keypoints", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 保存为 .npy 文件
X_data = np.array(X_data)
y_data = np.array(y_data)

print("本次采集完成，当前批次样本数量:", len(X_data))

# 如果已经有旧的数据文件，先加载再拼接
if os.path.exists("X_keypoints.npy") and os.path.exists("y_labels.npy"):
    print("检测到已有数据集，正在追加...")
    X_old = np.load("X_keypoints.npy")
    y_old = np.load("y_labels.npy")

    X_all = np.concatenate([X_old, X_data], axis=0)
    y_all = np.concatenate([y_old, y_data], axis=0)
else:
    print("未检测到旧数据集，本次为首次采集")
    X_all = X_data
    y_all = y_data

np.save("X_keypoints.npy", X_all)
np.save("y_labels.npy", y_all)

print("已保存为 X_keypoints.npy 和 y_labels.npy")
print("当前数据集总样本数量:", len(X_all))
print("标签种类:", np.unique(y_all))

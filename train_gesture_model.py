import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1. 加载训练数据（X = 63维关键点，y = 手势标签）
# -----------------------------
print("Loading dataset...")

# 请提前准备以下文件：
#   X_keypoints.npy    shape = (N, 63)
#   y_labels.npy       shape = (N, ) 取值为 {0,1,2}

X = np.load("X_keypoints.npy")
y = np.load("y_labels.npy")

print(f"Dataset Loaded: X={X.shape}, y={y.shape}")


# -----------------------------
# 2. 划分训练集/测试集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test Split Done.")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# -----------------------------
# 3. 定义 MLP 分类模型
# -----------------------------
# 模型结构：63 → 128 → 64 → 3
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    batch_size=32,
    shuffle=True,
    random_state=42
)


# -----------------------------
# 4. 开始训练
# -----------------------------
print("Training Model...")
model.fit(X_train, y_train)
print("Training Finished!")


# -----------------------------
# 5. 评估模型
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n===== TEST ACCURACY =====")
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 6. 保存模型和权重
# -----------------------------
print("\nSaving model...")
joblib.dump(model, "gesture_mlp.pkl")
print("Model saved as gesture_mlp.pkl")

# 导出各层权重用于 C++ 推理（可选）
np.save("fc1_w.npy", model.coefs_[0])
np.save("fc1_b.npy", model.intercepts_[0])

np.save("fc2_w.npy", model.coefs_[1])
np.save("fc2_b.npy", model.intercepts_[1])

np.save("fc3_w.npy", model.coefs_[2])
np.save("fc3_b.npy", model.intercepts_[2])

print("Weights exported as fc*_*.npy")

print("\nAll Done!")


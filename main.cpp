#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

#include <opencv4/opencv2/opencv.hpp>
#include <zmq.h>         // C API
#include <nlohmann/json.hpp> // 如果没有这个库，我后面给你去掉 JSON 方案

using json = nlohmann::json;
using namespace std;
using namespace cv;

// -------------------- 手势类型定义 --------------------
enum class GestureType {
    NONE = -1,
    WAVE = 0,       // 播放/暂停
    FIST = 1,       // 音量减小
    OPEN_PALM = 2   // 音量增大
};

string gestureTypeToString(GestureType g) {
    switch (g) {
        case GestureType::WAVE:       return "WAVE (Play/Pause)";
        case GestureType::FIST:       return "FIST (Volume Down)";
        case GestureType::OPEN_PALM:  return "OPEN_PALM (Volume Up)";
        default:                      return "None";
    }
}

// -------------------- 简单矩阵结构 --------------------
struct Matrix {
    int rows;
    int cols;
    vector<double> data; // 行优先

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& operator()(int r, int c) {
        return data[r * cols + c];
    }

    const double& operator()(int r, int c) const {
        return data[r * cols + c];
    }
};

// -------------------- CSV 读入 --------------------
bool loadCSV(const string& path, Matrix& mat) {
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "[ERROR] Cannot open file: " << path << endl;
        return false;
    }

    vector<vector<double>> temp;
    string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string item;
        vector<double> row;
        while (std::getline(ss, item, ',')) {
            try {
                row.push_back(std::stod(item));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) temp.push_back(row);
    }
    fin.close();

    if (temp.empty()) {
        cerr << "[ERROR] Empty CSV: " << path << endl;
        return false;
    }

    int r = (int)temp.size();
    int c = (int)temp[0].size();
    mat = Matrix(r, c);
    for (int i = 0; i < r; ++i) {
        if ((int)temp[i].size() != c) {
            cerr << "[ERROR] Inconsistent column size in: " << path << endl;
            return false;
        }
        for (int j = 0; j < c; ++j) {
            mat(i, j) = temp[i][j];
        }
    }
    cout << "[INFO] Loaded " << path << " shape = (" << r << ", " << c << ")" << endl;
    return true;
}

// -------------------- MLP 推理类 --------------------
// -------------------- MLP 推理类（适配 sklearn 的 (in, out) 形状） --------------------
class GestureMLP {
public:
    // W1: (63, hidden1), W2: (hidden1, hidden2), W3: (hidden2, out_dim)
    Matrix W1, W2, W3;
    vector<double> b1, b2, b3;

    bool loadWeights(const string& dir) {
        if (!loadCSV(dir + "/fc1_w.csv", W1)) return false;
        if (!loadCSV(dir + "/fc2_w.csv", W2)) return false;
        if (!loadCSV(dir + "/fc3_w.csv", W3)) return false;

        if (!loadBias(dir + "/fc1_b.csv", b1)) return false;
        if (!loadBias(dir + "/fc2_b.csv", b2)) return false;
        if (!loadBias(dir + "/fc3_b.csv", b3)) return false;

        cout << "[INFO] Loaded weights/fc1_w.csv shape = (" << W1.rows << ", " << W1.cols << ")" << endl;
        cout << "[INFO] Loaded weights/fc2_w.csv shape = (" << W2.rows << ", " << W2.cols << ")" << endl;
        cout << "[INFO] Loaded weights/fc3_w.csv shape = (" << W3.rows << ", " << W3.cols << ")" << endl;
        cout << "[INFO] Loaded weights/fc1_b.csv size = " << b1.size() << endl;
        cout << "[INFO] Loaded weights/fc2_b.csv size = " << b2.size() << endl;
        cout << "[INFO] Loaded weights/fc3_b.csv size = " << b3.size() << endl;

        // 这里按 sklearn 的 (in, out) 规则检查
        if (W1.rows != 63) {
            cerr << "[ERROR] W1.rows (输入维度) 应该是 63, 实际为 " << W1.rows << endl;
            return false;
        }
        if ((int)b1.size() != W1.cols) {
            cerr << "[ERROR] b1 大小应等于 W1.cols (隐藏层1神经元个数)." << endl;
            return false;
        }
        if (W2.rows != W1.cols) {
            cerr << "[ERROR] W2.rows 应等于 W1.cols (hidden1), 实际为 " << W2.rows << endl;
            return false;
        }
        if ((int)b2.size() != W2.cols) {
            cerr << "[ERROR] b2 大小应等于 W2.cols (隐藏层2神经元个数)." << endl;
            return false;
        }
        if (W3.rows != W2.cols) {
            cerr << "[ERROR] W3.rows 应等于 W2.cols (hidden2), 实际为 " << W3.rows << endl;
            return false;
        }
        if ((int)b3.size() != W3.cols) {
            cerr << "[ERROR] b3 大小应等于 W3.cols (输出维度/类别数)." << endl;
            return false;
        }

        cout << "[INFO] All weights loaded successfully." << endl;
        return true;
    }

    // 输入: 63 维特征（已经做过归一化）
    // 返回: (预测类别索引, 各类别 softmax 概率)
    pair<int, vector<double>> forward(const vector<double>& x) const {
        if ((int)x.size() != W1.rows) {
            cerr << "[WARN] input size " << x.size() << " != " << W1.rows << endl;
            return {-1, {}};
        }

        // ---- layer1: x (1, in_dim) * W1 (in_dim, hidden1) + b1 ----
        int hidden1 = W1.cols;
        vector<double> z1(hidden1, 0.0);
        for (int i = 0; i < hidden1; ++i) {          // 遍历输出神经元
            double s = 0.0;
            for (int j = 0; j < W1.rows; ++j) {      // 遍历输入维度
                s += x[j] * W1(j, i);
            }
            s += b1[i];
            z1[i] = std::max(0.0, s);                // ReLU
        }

        // ---- layer2: z1 (1, hidden1) * W2 (hidden1, hidden2) + b2 ----
        int hidden2 = W2.cols;
        vector<double> z2(hidden2, 0.0);
        for (int i = 0; i < hidden2; ++i) {
            double s = 0.0;
            for (int j = 0; j < W2.rows; ++j) {      // W2.rows == hidden1
                s += z1[j] * W2(j, i);
            }
            s += b2[i];
            z2[i] = std::max(0.0, s);                // ReLU
        }

        // ---- output layer: z2 (1, hidden2) * W3 (hidden2, out_dim) + b3 ----
        int out_dim = W3.cols;
        vector<double> logits(out_dim, 0.0);
        for (int i = 0; i < out_dim; ++i) {
            double s = 0.0;
            for (int j = 0; j < W3.rows; ++j) {      // W3.rows == hidden2
                s += z2[j] * W3(j, i);
            }
            s += b3[i];
            logits[i] = s;
        }

        // softmax（即使 out_dim == 1 也能正常工作，概率就是 1）
        vector<double> probs = softmax(logits);

        int argmax = 0;
        double best = probs[0];
        for (int i = 1; i < (int)probs.size(); ++i) {
            if (probs[i] > best) {
                best = probs[i];
                argmax = i;
            }
        }

        return {argmax, probs};
    }

private:
    bool loadBias(const string& path, vector<double>& b) {
        Matrix mat;
        if (!loadCSV(path, mat)) return false;
        if (mat.rows != 1) {
            cerr << "[ERROR] Bias file " << path << " 应该是 1 行, 实际为 " << mat.rows << endl;
            return false;
        }
        b.resize(mat.cols);
        for (int j = 0; j < mat.cols; ++j) {
            b[j] = mat(0, j);
        }
        return true;
    }

    static vector<double> softmax(const vector<double>& v) {
        vector<double> res(v.size());
        double maxv = *max_element(v.begin(), v.end());
        double sum = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            res[i] = std::exp(v[i] - maxv);
            sum += res[i];
        }
        for (double& x : res) {
            x /= (sum + 1e-10);
        }
        return res;
    }
};


// -------------------- 关键点归一化 --------------------
// 参考你在报告里写的“改进关键点归一化算法，减少手部大小影响”。:contentReference[oaicite:6]{index=6}
vector<double> normalizeKeypoints(const vector<double>& kp) {
    // kp size = 63 = 21 * 3
    if (kp.size() != 63) return {};

    // 以第 0 个关键点（一般为手腕）为中心进行平移，并按最大距离缩放
    double cx = kp[0];
    double cy = kp[1];
    double cz = kp[2];

    vector<double> out(63);
    double max_dist = 1e-8;
    for (int i = 0; i < 21; ++i) {
        double x = kp[i * 3 + 0] - cx;
        double y = kp[i * 3 + 1] - cy;
        double z = kp[i * 3 + 2] - cz;
        out[i * 3 + 0] = x;
        out[i * 3 + 1] = y;
        out[i * 3 + 2] = z;
        double d = std::sqrt(x * x + y * y + z * z);
        if (d > max_dist) max_dist = d;
    }
    // 归一化到 [-1,1] 区间
    for (double& v : out) {
        v /= max_dist;
    }
    return out;
}

// -------------------- ZMQ 接收 + JSON 解析 --------------------
// 这里直接使用 ZeroMQ C API 和 nlohmann::json 解析 python 发送的 JSON 数组。
bool recvLandmarks63(void* socket, vector<double>& out_kp) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket, 0);
    if (rc == -1) {
        zmq_msg_close(&msg);
        return false;
    }

    string payload(static_cast<char*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
    zmq_msg_close(&msg);

    try {
        auto j = json::parse(payload);
        if (!j.is_array()) {
            out_kp.clear();
            return true; // 空或非法就当没手
        }
        out_kp.clear();
        for (auto& v : j) {
            out_kp.push_back(v.get<double>());
        }
        return true;
    } catch (const std::exception& e) {
        cerr << "[ERROR] JSON parse failed: " << e.what() << endl;
        out_kp.clear();
        return false;
    }
}

// -------------------- 主函数 --------------------
int main() {
    // 1. 初始化 MLP
    GestureMLP mlp;
    if (!mlp.loadWeights("weights")) {
        cerr << "[FATAL] Failed to load weights, exit." << endl;
        return -1;
    }

    // 2. 初始化 ZeroMQ PULL（接收 python 发送的 63 维数据）
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PULL);
    // 注意这里要与 python 中的 bind/connect 对应：
    // 如果 python 是 socket.bind("tcp://*:5555")，这里就要 connect
    int rc = zmq_connect(socket, "tcp://localhost:5555");
    if (rc != 0) {
        cerr << "[FATAL] zmq_connect failed." << endl;
        zmq_close(socket);
        zmq_ctx_term(context);
        return -1;
    }

    // 3. 打开一个简单的 OpenCV 窗口（只是展示文字，不再自己采集摄像头）
    const int width = 640;
    const int height = 480;

    // 最近 N 帧预测，用于平滑
    const int SMOOTH_WINDOW = 5;
    deque<int> pred_history; // 存储最近 N 帧的预测标签

    cout << "C++ Gesture Receiver & Classifier started." << endl;
    cout << "Press ESC in the window to exit." << endl;

    while (true) {
        vector<double> kp;
        if (!recvLandmarks63(socket, kp)) {
            cerr << "[WARN] Failed to recv landmarks, continue." << endl;
            continue;
        }

        GestureType gesture = GestureType::NONE;
        double confidence = 0.0;
        if (kp.size() == 63) {
            // 归一化
            vector<double> norm_kp = normalizeKeypoints(kp);
            auto [pred_label, probs] = mlp.forward(norm_kp);
            if (pred_label >= 0 && pred_label <= 2) {
                // 更新平滑队列
                pred_history.push_back(pred_label);
                if ((int)pred_history.size() > SMOOTH_WINDOW) {
                    pred_history.pop_front(); // 或 pop_front，根据你想让“最近”优先还是“最旧”优先
                }

                // 基于输出维度自动设置类别数量
                int num_classes = probs.size();

                // 多数投票的计数容器
                vector<int> count(num_classes, 0);

                // 累计历史窗口的投票
                for (int v : pred_history) {
                    if (0 <= v && v < num_classes) count[v]++;
                }

                // 选出得票最多的类别
                int vote_label = 0;
                int best_cnt = count[0];
                for (int i = 1; i < num_classes; ++i) {
                    if (count[i] > best_cnt) {
                        best_cnt = count[i];
                        vote_label = i;
                    }
                }

                // 取投票结果对应的概率（大致代表置信度）
                confidence = (vote_label < (int)probs.size()) ? probs[vote_label] : 0.0;

                if (confidence > 0.6) { // 简单阈值，避免乱跳
                    if (vote_label == 0) gesture = GestureType::WAVE;
                    else if (vote_label == 1) gesture = GestureType::FIST;
                    else if (vote_label == 2) gesture = GestureType::OPEN_PALM;
                } else {
                    gesture = GestureType::NONE;
                }
            }
        } else {
            // 没检测到手
            pred_history.clear();
            gesture = GestureType::NONE;
        }

        // 4. 绘制 UI
        Mat canvas(height, width, CV_8UC3, Scalar(30, 30, 30));
        string gtext = "Gesture: " + gestureTypeToString(gesture);
        string ctext = "Confidence: " + to_string(confidence);

        putText(canvas, "Hand Gesture Recognition (C++ MLP)",
                Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 255, 255), 2);

        putText(canvas, gtext, Point(20, 120),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        putText(canvas, ctext, Point(20, 180),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

        // 简单给出功能提示
        putText(canvas, "WAVE: Play/Pause",
                Point(20, 260), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
        putText(canvas, "FIST: Volume Down",
                Point(20, 300), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
        putText(canvas, "OPEN_PALM: Volume Up",
                Point(20, 340), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);

        imshow("Gesture UI", canvas);
        int key = waitKey(1);
        if (key == 27) { // ESC
            break;
        }
    }

    zmq_close(socket);
    zmq_ctx_term(context);
    destroyAllWindows();
    return 0;
}

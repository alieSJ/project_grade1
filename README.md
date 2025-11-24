大一年度项目<br>
<title>一.手势采集<br></title>
1.激活虚拟环境<br>
    source ~/gesture_venv/bin/activate
    cd ~/project_grade1
2.运行采集脚本<br>
    python collect_keypoints.py
3.重复运行脚本三次可以采集三组不同数据<br>
<title>二.重新训练模型并导出权重</title>
1.运行以下代码<br>
    source ~/gesture_venv/bin/activate
    cd ~/project_grade1
    python train_gesture_model.py
    python save_npy_to_csv.py
train_gesture_model.py 会重新用新数据训练 63→128→64→3 的 MLP，并保存成 gesture_mlp.pkl 和 fc*_*.npy。<br>
save_npy_to_csv.py 会把这些 npy 导出为 weights/fc*_*.csv。<br>
<title>三.运行程序（需要两个终端）</title>
1.python端（发送关键点）<br>
    source ~/gesture_venv/bin/activate
    cd ~/gesture_project
    python mediapipe_sender.py
2.C++端（接收 + 分类 + 显示 UI）
    cd ~/gesture_project/build
    ./gesture_recognition


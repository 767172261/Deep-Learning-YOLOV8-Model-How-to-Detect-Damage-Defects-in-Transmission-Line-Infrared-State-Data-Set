输电线路红外状态下损伤检测数据集，共2393张，使用YOLO格式标注。可直接进行模型训练。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1cd51625e3b24362ad8ac8012a2daef2.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5f0a0fead9d84dfdbb83778ff6369edb.png)
 **输电线路红外状态下损伤检测数据集**（2393 张图像，YOLO 格式标注）构建完整 **YOLOv8 损伤检测系统** 的详细指南，包含：

- ✅ 数据准备  
- ✅ YOLOv8 训练代码（调用预训练模型）  
- ✅ 推理（单图/批量/视频）  
- ✅ 简易图形界面（PyQt5 + OpenCV）  

---

## 📌 一、数据集信息

| 项目 | 内容 |
|------|------|
| 图像数量 | 2393 张 |
| 模态 | 红外（单通道灰度图） |
| 标注格式 | YOLO 格式（`.txt`） |
| 任务类型 | 目标检测（如：绝缘子破损、导线过热、金具异常等） |
| 类别数 | 假设为 1 类 `damage`（可根据实际修改） |

> 若有多个类别，请在 `dataset.yaml` 中更新 `nc` 和 `names`。

---

## 🗂️ 二、数据目录结构

```
ir_damage_dataset/
├── images/
│   └── train/
│       ├── img001.jpg
│       └── ...
├── labels/
│   └── train/
│       ├── img001.txt
│       └── ...
└── dataset.yaml
```

### `dataset.yaml` 示例（单类别）

```yaml
# dataset.yaml
path: /your/path/to/ir_damage_dataset
train: images/train
val: images/train   # 建议后续划分验证集

nc: 1
names: ['damage']
```

> 💡 **建议**：从 2393 张中随机抽取 10%~20% 作为 `val`，提升评估可靠性。

---

## 🧪 三、YOLOv8 完整训练代码（Python）

```python
# train_ir_damage.py
from ultralytics import YOLO
import torch

# 1. 加载预训练模型（YOLOv8n/s/m）
model = YOLO('yolov8n.pt')  # 可换为 yolov8s.pt 等

# 2. 修改输入通道（红外图为单通道 → 需扩展为3通道或修改首层）
# 方案：将单通道复制为3通道（简单有效）
# （无需改模型结构，推理时自动处理）

# 3. 开始训练
results = model.train(
    data='ir_damage_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,          # 输入尺寸（YOLOv8 默认640，可设为416/512）
    batch=16,           # 根据GPU显存调整（16 for 16GB GPU）
    name='ir_damage_yolov8n',
    project='runs/ir_damage',
    device=0,           # GPU ID，CPU 用 'cpu'
    cache=False,        # 缓存图像加速（占内存）
    workers=8,
    exist_ok=True,
    verbose=True
)

print("✅ 训练完成！模型保存在:", results.save_dir)
```

> 🔔 **红外图处理说明**：  
> 虽然红外图是单通道，但 YOLOv8 默认接受 3 通道 RGB。  
> **解决方法**：在推理前将灰度图转为 3 通道（`cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)`），无需修改模型。

---

## 🔍 四、推理代码（单图 / 视频 / 批量）

```python
# infer_ir_damage.py
import cv2
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/ir_damage/ir_damage_yolov8n/weights/best.pt')

def detect_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转3通道
    results = model(img_bgr, conf=0.3)  # 置信度阈值
    annotated = results[0].plot()  # 绘制检测框
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例：检测单张图
detect_single_image("test_ir.jpg")
```

### 视频流实时检测（无人机/监控）

```python
cap = cv2.VideoCapture(0)  # 或视频文件路径
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    results = model(frame_bgr)
    annotated = results[0].plot()
    cv2.imshow("IR Damage Detection", annotated)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## 🖥️ 五、简易图形界面（PyQt5 + OpenCV）

安装依赖：
```bash
pip install pyqt5 opencv-python ultralytics numpy
```

```python
# gui_ir_detector.py
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import numpy as np

class IRDamageDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("输电线路红外损伤检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.model = YOLO('runs/ir_damage/ir_damage_yolov8n/weights/best.pt')
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.load_btn = QPushButton("加载红外图像")
        self.load_btn.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.current_image = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择红外图像", "", "Images (*.jpg *.png *.bmp)")
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            results = self.model(img_bgr, conf=0.3)
            annotated = results[0].plot()

            # 转为 QImage 显示
            rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(700, 500, Qt.KeepAspectRatio))

app = QApplication(sys.argv)
window = IRDamageDetector()
window.show()
sys.exit(app.exec_())
```

运行界面：
```bash
python gui_ir_detector.py
```

---

## 🚀 六、部署建议

| 场景 | 方案 |
|------|------|
| **无人机巡检** | Jetson Nano/Xavier + TensorRT 加速 |
| **变电站监控** | RTSP 视频流 + Flask API 后台 |
| **离线分析** | 批量推理脚本 + 生成检测报告（CSV/JSON） |
| **边缘设备** | 导出 ONNX → OpenVINO / NCNN |

导出 ONNX：
```python
model.export(format='onnx', imgsz=640, dynamic=True)
```

---



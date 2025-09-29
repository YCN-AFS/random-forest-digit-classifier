# MNIST Handwritten Digit Recognition - Hybrid Java+Python

Ứng dụng kết hợp Java Swing GUI với Python TensorFlow CNN để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST.

## 🚀 Kiến trúc Hybrid

```
┌─────────────────────────────────────────┐
│              Java GUI                   │
│  - Swing Interface                      │
│  - Data Loading (Weka)                  │
│  - File Management                      │
│  - Visualization                        │
└─────────────────┬───────────────────────┘
                  │ Process Communication
┌─────────────────▼───────────────────────┐
│            Python Backend               │
│  - TensorFlow/Keras CNN                 │
│  - Model Training                       │
│  - Prediction                           │
│  - Evaluation                           │
└─────────────────────────────────────────┘
```

## ✨ Tính năng chính

- **Java Swing GUI**: Giao diện thân thiện, dễ sử dụng
- **Python CNN**: Sử dụng TensorFlow/Keras cho deep learning
- **Weka Integration**: Xử lý dữ liệu CSV mạnh mẽ
- **Real-time Prediction**: Dự đoán nhanh chóng
- **Visualization**: Confusion matrix, biểu đồ trực quan
- **Save/Load**: Lưu trữ mô hình và kết quả

## 🧠 Kiến trúc CNN

```
Input (28, 28, 1)
    ↓
Conv2D (32 filters, 3x3) + ReLU + BatchNorm
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2) + Dropout(0.25)
    ↓
Conv2D (64 filters, 3x3) + ReLU + BatchNorm
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2) + Dropout(0.25)
    ↓
Conv2D (128 filters, 3x3) + ReLU + BatchNorm
    ↓
Dropout(0.25)
    ↓
Flatten
    ↓
Dense(512) + ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense(10) + Softmax
```

## 📋 Yêu cầu hệ thống

### Java
- **JDK**: 8 trở lên
- **JRE**: 8 trở lên
- **Weka**: 3.8.6

### Python
- **Python**: 3.8 trở lên
- **TensorFlow**: 2.10.0 trở lên
- **NumPy**: 1.21.0 trở lên
- **Pandas**: 1.3.0 trở lên
- **Scikit-learn**: 1.0.0 trở lên

### Hệ thống
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Storage**: 2GB trống
- **OS**: Windows, macOS, Linux

## 🛠️ Cài đặt và chạy

### 1. Tải dữ liệu MNIST

Tải file CSV từ Kaggle:
- [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- Đặt file `mnist_train.csv` và `mnist_test.csv` vào thư mục `archive/`

### 2. Cài đặt dependencies

```bash
# Cấp quyền thực thi
chmod +x build_hybrid.sh run_hybrid.sh

# Cài đặt và biên dịch (tự động cài Python dependencies)
./build_hybrid.sh
```

### 3. Chạy chương trình

```bash
./run_hybrid.sh
```

## 🎯 Hướng dẫn sử dụng

### Bước 1: Load dữ liệu
1. Nhấn **"Chọn file Train"** → Chọn `mnist_train.csv`
2. Nhấn **"Chọn file Test"** → Chọn `mnist_test.csv`

### Bước 2: Huấn luyện mô hình
1. Nhấn **"Huấn luyện CNN (Python)"**
2. Chờ quá trình huấn luyện hoàn thành (10-20 phút)
3. Python sẽ tự động tạo CNN model

### Bước 3: Đánh giá mô hình
1. Nhấn **"Đánh giá"**
2. Xem kết quả accuracy và confusion matrix

### Bước 4: Dự đoán
1. Nhập số dòng test (0-9999)
2. Nhấn **"Dự đoán"**
3. Xem kết quả dự đoán và độ tin cậy

### Bước 5: Xem biểu đồ
1. Nhấn **"Hiển thị biểu đồ"**
2. Xem confusion matrix heatmap

### Bước 6: Lưu trữ
- **"Lưu mô hình"**: Lưu CNN model
- **"Load mô hình"**: Tải CNN model đã lưu
- **"Lưu kết quả"**: Lưu kết quả đánh giá
- **"Lưu biểu đồ"**: Lưu biểu đồ dưới dạng PNG
- **"Lưu tất cả"**: Lưu tất cả cùng lúc

## 📊 Kết quả mong đợi

- **Accuracy**: 95-98% (cao hơn Random Forest)
- **Thời gian training**: 10-20 phút
- **Memory usage**: 4-8GB RAM
- **Confusion Matrix**: Ma trận 10x10 với đường chéo chính màu xanh

## 🔧 Cấu hình CNN

```python
# Tham số có thể điều chỉnh trong mnist_cnn.py
EPOCHS = 10              # Số epoch
BATCH_SIZE = 128         # Kích thước batch
LEARNING_RATE = 0.001    # Tốc độ học
DROPOUT_RATE = 0.25      # Tỷ lệ dropout
```

## 📁 Cấu trúc thư mục

```
java-v2/
├── MLGuiAppHybrid.java      # Java GUI chính
├── mnist_cnn.py             # Python CNN script
├── requirements.txt         # Python dependencies
├── build_hybrid.sh          # Script biên dịch
├── run_hybrid.sh            # Script chạy
├── README_HYBRID.md         # Hướng dẫn này
├── weka.jar                 # Weka library
├── archive/                 # Dữ liệu MNIST
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── *.class                  # File compiled
└── cnn_model_*              # Model files (tự tạo)
```

## 🆚 So sánh với các phiên bản khác

| Tính năng | Random Forest | Neural Network | Hybrid CNN |
|-----------|---------------|----------------|------------|
| **Accuracy** | 85-92% | 90-95% | 95-98% |
| **Thời gian train** | 5-10 giây | 2-5 phút | 10-20 phút |
| **Bộ nhớ** | 50MB | 200MB | 4-8GB |
| **Xử lý ảnh** | Pixel features | Hidden layers | Spatial features |
| **Khả năng mở rộng** | Hạn chế | Tốt | Rất tốt |
| **Dependencies** | Chỉ Weka | Chỉ Weka | Java + Python |

## 🐛 Xử lý lỗi thường gặp

### 1. Python dependencies không cài được
```bash
# Cài đặt thủ công
pip3 install tensorflow numpy pandas scikit-learn
```

### 2. OutOfMemoryError
```bash
# Tăng heap memory
java -Xmx8g -cp "weka.jar:." MLGuiAppHybrid
```

### 3. Python script không chạy được
```bash
# Kiểm tra Python version
python3 --version

# Kiểm tra dependencies
python3 -c "import tensorflow as tf; print('OK')"
```

### 4. Weka không tìm thấy
```bash
# Tải weka.jar
wget https://prdownloads.sourceforge.net/weka/weka-3-8-6.zip
unzip weka-3-8-6.zip
cp weka-3-8-6/weka.jar .
```

## 📈 Hiệu suất

- **Training time**: 10-20 phút (10 epochs)
- **Memory usage**: 4-8GB RAM
- **Model size**: 50-100MB
- **Prediction time**: <1 giây
- **Accuracy**: 95-98%

## 🔮 Hướng phát triển

### Ngắn hạn
- **Data Augmentation**: Xoay, zoom, shift ảnh
- **Transfer Learning**: Sử dụng pre-trained models
- **Ensemble Methods**: Kết hợp nhiều CNN
- **Real-time Webcam**: Nhận dạng từ camera

### Dài hạn
- **Mobile App**: Android/iOS version
- **Web Service**: REST API
- **Cloud Deployment**: AWS/Azure
- **Edge Computing**: Raspberry Pi

## 📚 Tài liệu tham khảo

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Weka Documentation](https://www.cs.waikato.ac.nz/ml/weka/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## 👥 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**Lưu ý**: Phiên bản Hybrid này kết hợp ưu điểm của Java (GUI) và Python (Deep Learning) để tạo ra ứng dụng mạnh mẽ và dễ sử dụng.
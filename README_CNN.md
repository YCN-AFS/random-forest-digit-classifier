# MNIST Handwritten Digit Recognition - CNN Version

Ứng dụng Java Swing sử dụng Convolutional Neural Network (CNN) với Deeplearning4j để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST.

## 🚀 Tính năng chính

- **CNN Architecture**: Sử dụng Deeplearning4j với kiến trúc CNN hiện đại
- **Giao diện thân thiện**: Java Swing với 10 nút chức năng
- **Xử lý ảnh**: Chuyển đổi dữ liệu CSV thành ảnh 28x28 pixel
- **Huấn luyện mô hình**: CNN với 2 lớp Convolution + 2 lớp Pooling + Dense + Output
- **Đánh giá chi tiết**: Accuracy, Confusion Matrix, Per-class metrics
- **Trực quan hóa**: Biểu đồ confusion matrix
- **Lưu trữ**: Mô hình, kết quả, biểu đồ

## 🏗️ Kiến trúc CNN

```
Input (1, 28, 28) 
    ↓
Conv2D (20 filters, 5x5) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (50 filters, 5x5) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Dense (500 neurons) + ReLU
    ↓
Output (10 classes) + Softmax
```

## 📋 Yêu cầu hệ thống

- **Java**: JDK 8 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Dependencies**: Deeplearning4j, ND4J, DataVec, Weka

## 🛠️ Cài đặt và chạy

### 1. Tải dữ liệu MNIST

Tải file CSV từ Kaggle:
- [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- Đặt file `mnist_train.csv` và `mnist_test.csv` vào thư mục `archive/`

### 2. Biên dịch chương trình

```bash
# Cấp quyền thực thi
chmod +x build_cnn.sh run_cnn.sh

# Biên dịch (sẽ tự động tải dependencies)
./build_cnn.sh
```

### 3. Chạy chương trình

```bash
./run_cnn.sh
```

## 🎯 Hướng dẫn sử dụng

### Bước 1: Load dữ liệu
1. Nhấn **"Chọn file Train"** → Chọn `mnist_train.csv`
2. Nhấn **"Chọn file Test"** → Chọn `mnist_test.csv`

### Bước 2: Huấn luyện mô hình
1. Nhấn **"Huấn luyện CNN"**
2. Chờ quá trình huấn luyện hoàn thành (5-10 phút)

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
- **"Lưu mô hình"**: Lưu CNN đã train
- **"Load mô hình"**: Tải CNN đã lưu
- **"Lưu kết quả"**: Lưu kết quả đánh giá
- **"Lưu biểu đồ"**: Lưu biểu đồ dưới dạng PNG
- **"Lưu tất cả"**: Lưu tất cả cùng lúc

## 📊 Kết quả mong đợi

- **Accuracy**: 95-98% (cao hơn Random Forest)
- **Thời gian huấn luyện**: 5-10 phút
- **Confusion Matrix**: Ma trận 10x10 với đường chéo chính màu xanh
- **Per-class Accuracy**: Độ chính xác từng chữ số 0-9

## 🔧 Cấu hình CNN

```java
// Tham số có thể điều chỉnh
private static final int HEIGHT = 28;           // Chiều cao ảnh
private static final int WIDTH = 28;            // Chiều rộng ảnh
private static final int CHANNELS = 1;          // Số kênh (grayscale)
private static final int NUM_CLASSES = 10;      // Số lớp (0-9)
private static final int BATCH_SIZE = 64;       // Kích thước batch
private static final int EPOCHS = 10;           // Số epoch
```

## 📁 Cấu trúc thư mục

```
java-v2/
├── MLGuiAppCNN.java          # File chính
├── build_cnn.sh              # Script biên dịch
├── run_cnn.sh                # Script chạy
├── README_CNN.md             # Hướng dẫn này
├── lib/                      # Dependencies
│   ├── dl4j-core-*.jar
│   ├── nd4j-native-*.jar
│   ├── datavec-api-*.jar
│   └── weka.jar
├── archive/                  # Dữ liệu MNIST
│   ├── mnist_train.csv
│   └── mnist_test.csv
└── *.class                   # File compiled
```

## 🆚 So sánh với Random Forest

| Tính năng | Random Forest | CNN |
|-----------|---------------|-----|
| **Accuracy** | 85-92% | 95-98% |
| **Thời gian train** | 5-10 giây | 5-10 phút |
| **Bộ nhớ** | 50MB | 500MB+ |
| **Xử lý ảnh** | Pixel features | Spatial features |
| **Khả năng mở rộng** | Hạn chế | Rất tốt |

## 🐛 Xử lý lỗi thường gặp

### 1. OutOfMemoryError
```bash
# Tăng heap memory
java -Xmx8g -cp "$CLASSPATH" MLGuiAppCNN
```

### 2. Dependencies không tải được
```bash
# Xóa thư mục lib và tải lại
rm -rf lib/
./build_cnn.sh
```

### 3. Lỗi biên dịch
```bash
# Kiểm tra Java version
java -version
javac -version

# Cần JDK 8+
```

## 📈 Hiệu suất

- **Training time**: 5-10 phút (10 epochs)
- **Memory usage**: 4-8GB RAM
- **Model size**: 50-100MB
- **Prediction time**: <1 giây

## 🔮 Hướng phát triển

- **Data Augmentation**: Xoay, zoom, shift ảnh
- **Transfer Learning**: Sử dụng pre-trained models
- **Ensemble Methods**: Kết hợp nhiều CNN
- **Real-time Prediction**: Webcam input
- **Mobile App**: Android/iOS version

## 📚 Tài liệu tham khảo

- [Deeplearning4j Documentation](https://deeplearning4j.konduit.ai/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## 👥 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**Lưu ý**: Phiên bản CNN này yêu cầu nhiều tài nguyên hơn so với Random Forest, nhưng cho kết quả chính xác cao hơn đáng kể.
# MNIST Handwritten Digit Recognition - Java GUI App

Ứng dụng Java với giao diện Swing để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST CSV sử dụng thuật toán RandomForest từ thư viện Weka.

## Tính năng

- 🖼️ **Giao diện thân thiện**: Sử dụng Java Swing với layout rõ ràng
- 📊 **Xử lý dữ liệu MNIST**: Đọc và chuẩn hóa dữ liệu CSV từ Kaggle
- 🤖 **Machine Learning**: Sử dụng RandomForest classifier từ Weka
- 📈 **Đánh giá mô hình**: Hiển thị accuracy, confusion matrix và các metrics chi tiết
- 🔮 **Dự đoán chữ số**: Chọn dòng test để dự đoán và so sánh kết quả

## Yêu cầu hệ thống

- Java 8 hoặc cao hơn
- Weka library (weka.jar)
- Dữ liệu MNIST CSV (mnist_train.csv, mnist_test.csv)

## Cài đặt

### 1. Tải Weka library

Tải file `weka.jar` từ trang chủ Weka:
```
https://www.cs.waikato.ac.nz/ml/weka/downloading.html
```

Đặt file `weka.jar` vào thư mục dự án.

### 2. Tải dữ liệu MNIST

Tải dữ liệu MNIST CSV từ Kaggle:
```
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
```

Đặt các file `mnist_train.csv` và `mnist_test.csv` vào thư mục dự án.

### 3. Cấu trúc thư mục

```
java-v2/
├── MLGuiApp.java          # File chính của ứng dụng
├── weka.jar              # Thư viện Weka
├── mnist_train.csv       # Dữ liệu training
├── mnist_test.csv        # Dữ liệu test
├── build.sh              # Script biên dịch
├── run.sh                # Script chạy ứng dụng
└── README.md             # Hướng dẫn này
```

## Sử dụng

### Biên dịch chương trình

```bash
chmod +x build.sh
./build.sh
```

Hoặc biên dịch thủ công:
```bash
javac -cp weka.jar:. MLGuiApp.java
```

### Chạy ứng dụng

```bash
chmod +x run.sh
./run.sh
```

Hoặc chạy thủ công:
```bash
java -cp .:weka.jar MLGuiApp
```

## Hướng dẫn sử dụng giao diện

### 1. Load dữ liệu
- **Chọn file Train**: Chọn file `mnist_train.csv` để load dữ liệu training
- **Chọn file Test**: Chọn file `mnist_test.csv` để load dữ liệu test

### 2. Huấn luyện mô hình
- Nhấn **"Huấn luyện mô hình"** để train RandomForest classifier
- Quá trình training có thể mất vài phút tùy thuộc vào kích thước dữ liệu

### 3. Đánh giá mô hình
- Nhấn **"Đánh giá"** để tính toán độ chính xác trên tập test
- Kết quả bao gồm:
  - Accuracy tổng thể
  - Confusion matrix
  - Precision, Recall, F1-score cho từng lớp

### 4. Dự đoán chữ số
- Nhập số dòng test (0 đến số mẫu test - 1)
- Nhấn **"Dự đoán"** để xem kết quả dự đoán
- So sánh chữ số thực tế với chữ số dự đoán

## Cấu trúc dữ liệu

Dữ liệu MNIST CSV có cấu trúc:
- **Cột đầu tiên**: `label` (0-9) - nhãn chữ số
- **784 cột tiếp theo**: `1x1`, `1x2`, ..., `28x28` - giá trị pixel (0-255)

Dữ liệu được tự động chuẩn hóa về khoảng [0,1] trước khi training.

## Thuật toán

- **Classifier**: RandomForest từ Weka
- **Số cây**: 100
- **Features**: Sử dụng tất cả 784 pixel features
- **Preprocessing**: Normalization về khoảng [0,1]

## Troubleshooting

### Lỗi "Không tìm thấy weka.jar"
- Đảm bảo file `weka.jar` đã được tải và đặt đúng vị trí
- Kiểm tra quyền đọc file

### Lỗi "OutOfMemoryError"
- Tăng heap size Java: `java -Xmx2g -cp .:weka.jar MLGuiApp`
- Hoặc sử dụng tập dữ liệu nhỏ hơn

### Lỗi load CSV
- Kiểm tra định dạng file CSV
- Đảm bảo file không bị hỏng
- Kiểm tra encoding (UTF-8)

## Tác giả

AI Assistant - Phiên bản 1.0

## License

Dự án này được tạo ra cho mục đích học tập và nghiên cứu.# random-forest-digit-classifier
# random-forest-digit-classifier

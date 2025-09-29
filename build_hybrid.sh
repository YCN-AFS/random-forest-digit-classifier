#!/bin/bash

echo "=== MNIST Handwritten Digit Recognition - Hybrid Java+Python Build Script ==="
echo "Sử dụng Java Swing GUI + Python TensorFlow CNN"
echo ""

# Kiểm tra Java
if ! command -v javac &> /dev/null; then
    echo "❌ Java compiler (javac) không được tìm thấy!"
    echo "Vui lòng cài đặt Java Development Kit (JDK) 8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Java compiler: $(javac -version 2>&1)"

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 không được tìm thấy!"
    echo "Vui lòng cài đặt Python 3.8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Python: $(python3 --version)"

# Kiểm tra pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 không được tìm thấy!"
    echo "Vui lòng cài đặt pip3"
    exit 1
fi

echo "✅ Tìm thấy pip3: $(pip3 --version)"

# Kiểm tra weka.jar
if [ ! -f "weka.jar" ]; then
    echo "❌ File weka.jar không tồn tại!"
    echo "Vui lòng tải weka.jar và đặt trong thư mục hiện tại"
    echo "Hoặc chạy: wget https://prdownloads.sourceforge.net/weka/weka-3-8-6.zip"
    echo "Sau đó giải nén và copy weka.jar vào thư mục này"
    exit 1
fi

echo "✅ Tìm thấy weka.jar"

# Kiểm tra Python script
if [ ! -f "mnist_cnn.py" ]; then
    echo "❌ File mnist_cnn.py không tồn tại!"
    exit 1
fi

echo "✅ Tìm thấy mnist_cnn.py"

# Cài đặt Python dependencies
echo ""
echo "🐍 Cài đặt Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Lỗi cài đặt Python dependencies!"
    echo "Vui lòng kiểm tra kết nối internet và quyền cài đặt"
    exit 1
fi

echo "✅ Python dependencies đã được cài đặt"

# Biên dịch Java
echo ""
echo "🔨 Đang biên dịch MLGuiAppHybrid.java..."

javac -cp "weka.jar:." -encoding UTF-8 MLGuiAppHybrid.java

if [ $? -eq 0 ]; then
    echo "✅ Biên dịch thành công!"
    echo ""
    echo "📋 Các file đã tạo:"
    echo "   - MLGuiAppHybrid.class"
    echo "   - MLGuiAppHybrid\$*.class (inner classes)"
    echo ""
    echo "🚀 Để chạy chương trình:"
    echo "   ./run_hybrid.sh"
    echo ""
    echo "📚 Kiến trúc Hybrid:"
    echo "   - Java Swing: GUI và xử lý dữ liệu"
    echo "   - Python TensorFlow: CNN model"
    echo "   - Weka: Data loading và preprocessing"
    echo ""
    echo "🎯 Tính năng:"
    echo "   - CNN với 3 lớp Convolution + Dense"
    echo "   - Accuracy: 95-98%"
    echo "   - Real-time prediction"
    echo "   - Confusion matrix visualization"
else
    echo "❌ Lỗi biên dịch!"
    echo ""
    echo "🔍 Kiểm tra lỗi:"
    echo "1. Đảm bảo weka.jar có trong thư mục hiện tại"
    echo "2. Kiểm tra phiên bản Java (cần JDK 8+)"
    echo "3. Kiểm tra quyền ghi file"
    exit 1
fi
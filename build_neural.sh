#!/bin/bash

echo "=== MNIST Handwritten Digit Recognition - Neural Network Build Script ==="
echo "Sử dụng Weka MultilayerPerceptron (Neural Network)"
echo ""

# Kiểm tra Java
if ! command -v javac &> /dev/null; then
    echo "❌ Java compiler (javac) không được tìm thấy!"
    echo "Vui lòng cài đặt Java Development Kit (JDK) 8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Java compiler: $(javac -version 2>&1)"

# Kiểm tra weka.jar
if [ ! -f "weka.jar" ]; then
    echo "❌ File weka.jar không tồn tại!"
    echo "Vui lòng tải weka.jar và đặt trong thư mục hiện tại"
    echo "Hoặc chạy: wget https://prdownloads.sourceforge.net/weka/weka-3-8-6.zip"
    echo "Sau đó giải nén và copy weka.jar vào thư mục này"
    exit 1
fi

echo "✅ Tìm thấy weka.jar"

echo ""
echo "🔨 Đang biên dịch MLGuiAppNeural.java..."

# Biên dịch với weka.jar
javac -cp "weka.jar:." -encoding UTF-8 MLGuiAppNeural.java

if [ $? -eq 0 ]; then
    echo "✅ Biên dịch thành công!"
    echo ""
    echo "📋 Các file đã tạo:"
    echo "   - MLGuiAppNeural.class"
    echo "   - MLGuiAppNeural\$*.class (inner classes)"
    echo ""
    echo "🚀 Để chạy chương trình:"
    echo "   ./run_neural.sh"
    echo ""
    echo "📚 Sử dụng Weka MultilayerPerceptron:"
    echo "   - Hidden layers: 200, 100"
    echo "   - Learning rate: 0.3"
    echo "   - Momentum: 0.2"
    echo "   - Training epochs: 500"
else
    echo "❌ Lỗi biên dịch!"
    echo ""
    echo "🔍 Kiểm tra lỗi:"
    echo "1. Đảm bảo weka.jar có trong thư mục hiện tại"
    echo "2. Kiểm tra phiên bản Java (cần JDK 8+)"
    echo "3. Kiểm tra quyền ghi file"
    exit 1
fi
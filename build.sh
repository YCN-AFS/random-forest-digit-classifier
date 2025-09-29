#!/bin/bash

# Build script cho MNIST Handwritten Digit Recognition App
# Sử dụng Weka library

echo "=== MNIST Handwritten Digit Recognition - Build Script ==="
echo

# Kiểm tra xem weka.jar có tồn tại không
if [ ! -f "weka.jar" ]; then
    echo "❌ Không tìm thấy weka.jar!"
    echo "Vui lòng tải weka.jar từ: https://www.cs.waikato.ac.nz/ml/weka/downloading.html"
    echo "Đặt file weka.jar vào thư mục hiện tại."
    exit 1
fi

echo "✓ Tìm thấy weka.jar"

# Biên dịch chương trình
echo "🔄 Đang biên dịch MLGuiApp.java..."
javac -cp weka.jar:. MLGuiApp.java

if [ $? -eq 0 ]; then
    echo "✓ Biên dịch thành công!"
    echo
    echo "🚀 Để chạy chương trình, sử dụng lệnh:"
    echo "   java -cp .:weka.jar MLGuiApp"
    echo
    echo "📁 Hoặc chạy script run.sh:"
    echo "   ./run.sh"
else
    echo "❌ Lỗi biên dịch!"
    exit 1
fi
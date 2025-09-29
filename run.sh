#!/bin/bash

# Run script cho MNIST Handwritten Digit Recognition App

echo "=== MNIST Handwritten Digit Recognition - Run Script ==="
echo

# Kiểm tra xem weka.jar có tồn tại không
if [ ! -f "weka.jar" ]; then
    echo "❌ Không tìm thấy weka.jar!"
    echo "Vui lòng tải weka.jar từ: https://www.cs.waikato.ac.nz/ml/weka/downloading.html"
    echo "Đặt file weka.jar vào thư mục hiện tại."
    exit 1
fi

# Kiểm tra xem file .class có tồn tại không
if [ ! -f "MLGuiApp.class" ]; then
    echo "❌ Không tìm thấy MLGuiApp.class!"
    echo "Vui lòng chạy build.sh trước để biên dịch chương trình."
    exit 1
fi

echo "✓ Tìm thấy weka.jar và MLGuiApp.class"
echo "🚀 Đang khởi động ứng dụng..."
echo

# Chạy chương trình
java -cp .:weka.jar MLGuiApp
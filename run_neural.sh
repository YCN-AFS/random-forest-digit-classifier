#!/bin/bash

echo "=== MNIST Handwritten Digit Recognition - Neural Network Run Script ==="
echo "Sử dụng Weka MultilayerPerceptron (Neural Network)"
echo ""

# Kiểm tra Java
if ! command -v java &> /dev/null; then
    echo "❌ Java runtime không được tìm thấy!"
    echo "Vui lòng cài đặt Java Runtime Environment (JRE) 8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Java runtime: $(java -version 2>&1 | head -n 1)"

# Kiểm tra file class
if [ ! -f "MLGuiAppNeural.class" ]; then
    echo "❌ File MLGuiAppNeural.class không tồn tại!"
    echo "Vui lòng chạy build_neural.sh trước"
    exit 1
fi

# Kiểm tra weka.jar
if [ ! -f "weka.jar" ]; then
    echo "❌ File weka.jar không tồn tại!"
    echo "Vui lòng tải weka.jar và đặt trong thư mục hiện tại"
    exit 1
fi

echo "✅ Tìm thấy MLGuiAppNeural.class và weka.jar"
echo "🚀 Đang khởi động ứng dụng Neural Network..."

# Chạy chương trình
java -cp "weka.jar:." \
     -Xmx2g \
     -Dfile.encoding=UTF-8 \
     -Djava.awt.headless=false \
     MLGuiAppNeural

echo ""
echo "👋 Ứng dụng đã kết thúc"
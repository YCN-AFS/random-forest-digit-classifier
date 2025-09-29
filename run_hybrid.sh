#!/bin/bash

echo "=== MNIST Handwritten Digit Recognition - Hybrid Java+Python Run Script ==="
echo "Sử dụng Java Swing GUI + Python TensorFlow CNN"
echo ""

# Kiểm tra Java
if ! command -v java &> /dev/null; then
    echo "❌ Java runtime không được tìm thấy!"
    echo "Vui lòng cài đặt Java Runtime Environment (JRE) 8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Java runtime: $(java -version 2>&1 | head -n 1)"

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 không được tìm thấy!"
    echo "Vui lòng cài đặt Python 3.8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Python: $(python3 --version)"

# Kiểm tra file class
if [ ! -f "MLGuiAppHybrid.class" ]; then
    echo "❌ File MLGuiAppHybrid.class không tồn tại!"
    echo "Vui lòng chạy build_hybrid.sh trước"
    exit 1
fi

# Kiểm tra weka.jar
if [ ! -f "weka.jar" ]; then
    echo "❌ File weka.jar không tồn tại!"
    echo "Vui lòng tải weka.jar và đặt trong thư mục hiện tại"
    exit 1
fi

# Kiểm tra Python script
if [ ! -f "mnist_cnn.py" ]; then
    echo "❌ File mnist_cnn.py không tồn tại!"
    exit 1
fi

echo "✅ Tìm thấy MLGuiAppHybrid.class, weka.jar và mnist_cnn.py"

# Kiểm tra Python dependencies
echo "🔍 Kiểm tra Python dependencies..."
python3 -c "import tensorflow as tf; import numpy as np; import pandas as pd; print('✅ Python dependencies OK')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ Python dependencies chưa được cài đặt!"
    echo "Vui lòng chạy: pip3 install -r requirements.txt"
    exit 1
fi

echo "✅ Python dependencies OK"

echo "🚀 Đang khởi động ứng dụng Hybrid..."

# Chạy chương trình
java -cp "weka.jar:." \
     -Xmx4g \
     -Dfile.encoding=UTF-8 \
     -Djava.awt.headless=false \
     MLGuiAppHybrid

echo ""
echo "👋 Ứng dụng đã kết thúc"
#!/bin/bash

echo "=== MNIST Handwritten Digit Recognition - CNN Run Script ==="
echo "Sử dụng Deeplearning4j (DL4J) cho Convolutional Neural Network"
echo ""

# Kiểm tra Java
if ! command -v java &> /dev/null; then
    echo "❌ Java runtime không được tìm thấy!"
    echo "Vui lòng cài đặt Java Runtime Environment (JRE) 8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Java runtime: $(java -version 2>&1 | head -n 1)"

# Kiểm tra file class
if [ ! -f "MLGuiAppCNN.class" ]; then
    echo "❌ File MLGuiAppCNN.class không tồn tại!"
    echo "Vui lòng chạy build_cnn.sh trước"
    exit 1
fi

# Kiểm tra dependencies
if [ ! -d "lib" ]; then
    echo "❌ Thư mục lib không tồn tại!"
    echo "Vui lòng chạy build_cnn.sh trước"
    exit 1
fi

# Tạo classpath
CLASSPATH="."
for jar in lib/*.jar; do
    CLASSPATH="$CLASSPATH:$jar"
done

echo "✅ Tìm thấy MLGuiAppCNN.class và dependencies"
echo "🚀 Đang khởi động ứng dụng CNN..."

# Chạy chương trình
java -cp "$CLASSPATH" \
     -Xmx4g \
     -Dfile.encoding=UTF-8 \
     -Djava.awt.headless=false \
     MLGuiAppCNN

echo ""
echo "👋 Ứng dụng đã kết thúc"
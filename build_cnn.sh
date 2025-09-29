#!/bin/bash

echo "=== MNIST Handwritten Digit Recognition - CNN Build Script ==="
echo "Sử dụng Deeplearning4j (DL4J) cho Convolutional Neural Network"
echo ""

# Kiểm tra Java
if ! command -v javac &> /dev/null; then
    echo "❌ Java compiler (javac) không được tìm thấy!"
    echo "Vui lòng cài đặt Java Development Kit (JDK) 8 trở lên"
    exit 1
fi

echo "✅ Tìm thấy Java compiler: $(javac -version 2>&1)"

# Tạo thư mục lib nếu chưa có
mkdir -p lib

# Download Deeplearning4j dependencies nếu chưa có
echo ""
echo "📦 Kiểm tra dependencies..."

if [ ! -f "lib/dl4j-core-1.0.0-M2.1.jar" ]; then
    echo "🔄 Đang tải Deeplearning4j Core..."
    wget -O lib/dl4j-core-1.0.0-M2.1.jar "https://repo1.maven.org/maven2/org/deeplearning4j/deeplearning4j-core/1.0.0-M2.1/deeplearning4j-core-1.0.0-M2.1.jar"
fi

if [ ! -f "lib/nd4j-native-1.0.0-M2.1.jar" ]; then
    echo "🔄 Đang tải ND4J Native..."
    wget -O lib/nd4j-native-1.0.0-M2.1.jar "https://repo1.maven.org/maven2/org/nd4j/nd4j-native/1.0.0-M2.1/nd4j-native-1.0.0-M2.1.jar"
fi

if [ ! -f "lib/datavec-api-1.0.0-M2.1.jar" ]; then
    echo "🔄 Đang tải DataVec API..."
    wget -O lib/datavec-api-1.0.0-M2.1.jar "https://repo1.maven.org/maven2/org/datavec/datavec-api/1.0.0-M2.1/datavec-api-1.0.0-M2.1.jar"
fi

if [ ! -f "lib/weka.jar" ]; then
    echo "🔄 Đang tải Weka..."
    wget -O lib/weka-3-8-6.zip "https://prdownloads.sourceforge.net/weka/weka-3-8-6.zip"
    unzip -j lib/weka-3-8-6.zip "weka-3-8-6/weka.jar" -d lib/
    mv lib/weka.jar lib/weka.jar
    rm lib/weka-3-8-6.zip
    rm -rf lib/weka-3-8-6
fi

# Tạo classpath
CLASSPATH="."
for jar in lib/*.jar; do
    CLASSPATH="$CLASSPATH:$jar"
done

echo ""
echo "🔨 Đang biên dịch MLGuiAppCNN.java..."

# Biên dịch với tất cả dependencies
javac -cp "$CLASSPATH" -encoding UTF-8 MLGuiAppCNN.java

if [ $? -eq 0 ]; then
    echo "✅ Biên dịch thành công!"
    echo ""
    echo "📋 Các file đã tạo:"
    echo "   - MLGuiAppCNN.class"
    echo "   - MLGuiAppCNN\$*.class (inner classes)"
    echo ""
    echo "🚀 Để chạy chương trình:"
    echo "   ./run_cnn.sh"
    echo ""
    echo "📚 Dependencies đã tải:"
    ls -la lib/*.jar
else
    echo "❌ Lỗi biên dịch!"
    echo ""
    echo "🔍 Kiểm tra lỗi:"
    echo "1. Đảm bảo tất cả JAR files đã được tải đúng"
    echo "2. Kiểm tra phiên bản Java (cần JDK 8+)"
    echo "3. Kiểm tra quyền ghi file"
    exit 1
fi
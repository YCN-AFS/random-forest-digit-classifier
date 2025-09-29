#!/usr/bin/env python3
"""
Tạo dữ liệu MNIST mẫu để test chương trình Java
"""

import numpy as np
import pandas as pd
import os

def create_sample_mnist_data():
    """Tạo dữ liệu MNIST mẫu với cấu trúc CSV"""
    
    print("Tạo dữ liệu MNIST mẫu...")
    
    # Tạo dữ liệu training (1000 mẫu)
    n_train = 1000
    n_test = 200
    
    # Tạo labels ngẫu nhiên (0-9)
    train_labels = np.random.randint(0, 10, n_train)
    test_labels = np.random.randint(0, 10, n_test)
    
    # Tạo pixel data ngẫu nhiên (784 pixels = 28x28)
    train_pixels = np.random.randint(0, 256, (n_train, 784))
    test_pixels = np.random.randint(0, 256, (n_test, 784))
    
    # Tạo tên cột cho pixels
    pixel_columns = []
    for i in range(28):
        for j in range(28):
            pixel_columns.append(f"{i+1}x{j+1}")
    
    # Tạo DataFrame cho training
    train_data = {'label': train_labels}
    for i, col in enumerate(pixel_columns):
        train_data[col] = train_pixels[:, i]
    
    train_df = pd.DataFrame(train_data)
    
    # Tạo DataFrame cho test
    test_data = {'label': test_labels}
    for i, col in enumerate(pixel_columns):
        test_data[col] = test_pixels[:, i]
    
    test_df = pd.DataFrame(test_data)
    
    # Lưu file CSV
    train_df.to_csv('mnist_train.csv', index=False)
    test_df.to_csv('mnist_test.csv', index=False)
    
    print(f"✓ Đã tạo mnist_train.csv với {n_train} mẫu")
    print(f"✓ Đã tạo mnist_test.csv với {n_test} mẫu")
    print("✓ Dữ liệu sẵn sàng để test chương trình!")

if __name__ == "__main__":
    create_sample_mnist_data()
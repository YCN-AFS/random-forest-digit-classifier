#!/usr/bin/env python3
"""
MNIST CNN Model - Python Backend
Sử dụng TensorFlow/Keras để tạo CNN model cho nhận dạng chữ số MNIST
"""

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os
from datetime import datetime

class MNISTCNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.is_trained = False
        
    def load_data(self, train_path, test_path):
        """Load MNIST data from CSV files"""
        print(f"🔄 Loading training data from {train_path}...")
        train_df = pd.read_csv(train_path)
        
        print(f"🔄 Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        
        # Separate features and labels
        X_train = train_df.iloc[:, 1:].values.astype('float32')
        y_train = train_df.iloc[:, 0].values.astype('int32')
        X_test = test_df.iloc[:, 1:].values.astype('float32')
        y_test = test_df.iloc[:, 0].values.astype('int32')
        
        # Reshape to 28x28 images
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Normalize pixel values to 0-1
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"✅ Training data: {X_train.shape[0]} samples, {X_train.shape[1]}x{X_train.shape[2]} images")
        print(f"✅ Test data: {X_test.shape[0]} samples, {X_test.shape[1]}x{X_test.shape[2]} images")
        
        return (X_train, y_train), (X_test, y_test)
    
    def create_model(self):
        """Create CNN model architecture"""
        print("🧠 Creating CNN model...")
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ CNN model created successfully!")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        """Train the CNN model"""
        print(f"🌐 Starting training for {epochs} epochs...")
        print(f"📊 Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("✅ Training completed!")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        print("📊 Evaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'class_accuracy': class_accuracy.tolist(),
            'classification_report': classification_report(y_true_classes, y_pred_classes, output_dict=True)
        }
        
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"✅ Test Loss: {test_loss:.4f}")
        
        return results
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        return predicted_classes, confidence_scores, predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"✅ Model loaded from {filepath}")

def main():
    parser = argparse.ArgumentParser(description='MNIST CNN Training and Evaluation')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], required=True,
                       help='Mode: train, evaluate, or predict')
    parser.add_argument('--train_data', type=str, help='Path to training CSV file')
    parser.add_argument('--test_data', type=str, help='Path to test CSV file')
    parser.add_argument('--model_path', type=str, help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--predict_data', type=str, help='Data to predict (JSON format)')
    
    args = parser.parse_args()
    
    # Initialize CNN
    cnn = MNISTCNN()
    
    try:
        if args.mode == 'train':
            if not args.train_data or not args.test_data:
                print("❌ Error: --train_data and --test_data are required for training")
                sys.exit(1)
            
            # Load data
            (X_train, y_train), (X_test, y_test) = cnn.load_data(args.train_data, args.test_data)
            
            # Create and train model
            cnn.create_model()
            cnn.train(X_train, y_train, X_test, y_test, args.epochs, args.batch_size)
            
            # Evaluate
            results = cnn.evaluate(X_test, y_test)
            
            # Save model
            if args.model_path:
                cnn.save_model(args.model_path)
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Results saved to {args.output}")
            
            # Print summary
            print("\n📊 TRAINING SUMMARY:")
            print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Final Test Loss: {results['test_loss']:.4f}")
            
        elif args.mode == 'evaluate':
            if not args.test_data or not args.model_path:
                print("❌ Error: --test_data and --model_path are required for evaluation")
                sys.exit(1)
            
            # Load model
            cnn.load_model(args.model_path)
            
            # Load test data
            test_df = pd.read_csv(args.test_data)
            X_test = test_df.iloc[:, 1:].values.astype('float32')
            y_test = test_df.iloc[:, 0].values.astype('int32')
            X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
            y_test = keras.utils.to_categorical(y_test, 10)
            
            # Evaluate
            results = cnn.evaluate(X_test, y_test)
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Results saved to {args.output}")
            
        elif args.mode == 'predict':
            if not args.model_path or not args.predict_data:
                print("❌ Error: --model_path and --predict_data are required for prediction")
                sys.exit(1)
            
            # Load model
            cnn.load_model(args.model_path)
            
            # Load prediction data
            data = json.loads(args.predict_data)
            X = np.array(data['features']).reshape(-1, 28, 28, 1) / 255.0
            
            # Predict
            predicted_classes, confidence_scores, predictions = cnn.predict(X)
            
            # Prepare results
            results = {
                'predictions': predicted_classes.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'probabilities': predictions.tolist()
            }
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Predictions saved to {args.output}")
            
            # Print results
            for i, (pred, conf) in enumerate(zip(predicted_classes, confidence_scores)):
                print(f"Sample {i}: Predicted={pred}, Confidence={conf:.4f}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# AI_Based_Disease_Prediction_System

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Embedding

# 1. LOAD AND PREPROCESS DATA
# Replace the URL below with your raw file or a local path like 'disease_dataset.csv'
DATA_URL = "https://raw.githubusercontent.com/Kiruthika2006-devil/AI_Based_Disease_Prediction_System/main/disease_dataset.csv"

try:
    df = pd.read_csv(DATA_URL)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Could not load from URL, please ensure the path is correct. Error: {e}")
    # Fallback to local file if needed: df = pd.read_csv('disease_dataset.csv')

# Drop completely empty columns if they exist
df = df.dropna(axis=1, how='all')

# Assume the last column contains the disease names (e.g., 'prognosis' or 'disease')
X = df.iloc[:, :-1].values  # Features (Symptoms: 0s and 1s)
y = df.iloc[:, -1].values   # Target (Disease Labels)

# Encode categorical text labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Number of symptoms (features): {X.shape[1]}")
print(f"Number of target diseases (classes): {num_classes}")


# 2. MODEL 1: ARTIFICIAL NEURAL NETWORK (ANN)
print("\n--- Training Artificial Neural Network (ANN) ---")

ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])

ann_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the ANN
ann_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate ANN
ann_preds = np.argmax(ann_model.predict(X_test), axis=-1)
ann_acc = accuracy_score(y_test, ann_preds)
print(f"\nANN Test Accuracy: {ann_acc * 100:.2f}%")


# 3. MODEL 2: RECURRENT NEURAL NETWORK (RNN)
print("\n--- Training Recurrent Neural Network (RNN) ---")

# For RNN, data must be 3D: (samples, timesteps, features)
# We treat each symptom feature as a timestep with 1 element
X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

rnn_model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(X_train_rnn.shape[1], 1), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

rnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the RNN
rnn_model.fit(X_train_rnn, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1)

# Evaluate RNN
rnn_preds = np.argmax(rnn_model.predict(X_test_rnn), axis=-1)
rnn_acc = accuracy_score(y_test, rnn_preds)
print(f"\nRNN Test Accuracy: {rnn_acc * 100:.2f}%")

# 4. FINAL COMPARISON REPORT
print("\n=== Final Performance Comparison ===")
print(f"ANN Accuracy: {ann_acc * 100:.2f}%")
print(f"RNN Accuracy: {rnn_acc * 100:.2f}%")

print("\nDetailed ANN Classification Report:")
print(classification_report(y_test, ann_preds, target_names=label_encoder.classes_))

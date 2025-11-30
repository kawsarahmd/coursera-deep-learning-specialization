#!/usr/bin/env python
# coding: utf-8

"""
Convolutional Neural Networks: Application (PyTorch Version)

This notebook demonstrates:
- Creating a mood classifier using PyTorch nn.Sequential
- Building a ConvNet to identify sign language digits using PyTorch nn.Module
- Binary and multiclass classification with CNNs
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cnn_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
torch.manual_seed(1)

# ==================== 1. Load Happy House Dataset ====================

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape labels
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# ==================== 2. PyTorch Model Definitions ====================

class HappyModel(nn.Module):
    """
    Binary classification model for Happy House dataset
    Architecture: ZEROPAD -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    PyTorch uses NCHW format (batch, channels, height, width)
    vs Keras NHWC format (batch, height, width, channels)
    """
    def __init__(self):
        super(HappyModel, self).__init__()

        # ZeroPadding2D with padding 3
        self.pad = nn.ZeroPad2d(3)

        # Conv2D with 32 7x7 filters, stride 1
        # Input: (batch, 3, 70, 70) after padding
        # Output: (batch, 32, 64, 64)
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=7, stride=1, padding=0)

        # BatchNormalization
        self.bn0 = nn.BatchNorm2d(32)

        # ReLU activation
        self.relu = nn.ReLU()

        # MaxPooling2D with default parameters (2x2)
        # Output: (batch, 32, 32, 32)
        self.max_pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten
        self.flatten = nn.Flatten()

        # Dense layer with 1 unit and sigmoid activation
        # Input: 32 * 32 * 32 = 32768
        self.fc = nn.Linear(32 * 32 * 32, 1)

    def forward(self, x):
        # Convert from NHWC (Keras format) to NCHW (PyTorch format)
        # Input x shape: (batch, 64, 64, 3)
        x = x.permute(0, 3, 1, 2)  # -> (batch, 3, 64, 64)

        x = self.pad(x)            # -> (batch, 3, 70, 70)
        x = self.conv0(x)          # -> (batch, 32, 64, 64)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.max_pool0(x)      # -> (batch, 32, 32, 32)
        x = self.flatten(x)        # -> (batch, 32768)
        x = self.fc(x)             # -> (batch, 1)
        x = torch.sigmoid(x)

        return x


class ConvolutionalModel(nn.Module):
    """
    Multiclass classification model for SIGNS dataset
    Architecture: CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    """
    def __init__(self, input_shape, num_classes=6):
        super(ConvolutionalModel, self).__init__()

        # CONV2D: 8 filters 4x4, stride 1, padding 'same'
        # Input: (batch, 3, 64, 64)
        # Output: (batch, 8, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8,
                               kernel_size=4, stride=1, padding=2)  # padding=2 for 'same'

        # ReLU
        self.relu1 = nn.ReLU()

        # MAXPOOL: window 8x8, stride 8, padding 'same'
        # Output: (batch, 8, 8, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)

        # CONV2D: 16 filters 2x2, stride 1, padding 'same'
        # Output: (batch, 16, 8, 8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=2, stride=1, padding=0)  # padding=0 for 'same' with 2x2

        # ReLU
        self.relu2 = nn.ReLU()

        # MAXPOOL: window 4x4, stride 4, padding 'same'
        # Output: (batch, 16, 2, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # Flatten
        self.flatten = nn.Flatten()

        # Dense layer with 6 neurons and softmax activation
        # Input: 16 * 2 * 2 = 64
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convert from NHWC (Keras format) to NCHW (PyTorch format)
        # Input x shape: (batch, 64, 64, 3)
        x = x.permute(0, 3, 1, 2)  # -> (batch, 3, 64, 64)

        x = self.conv1(x)          # -> (batch, 8, 64, 64)
        x = self.relu1(x)
        x = self.pool1(x)          # -> (batch, 8, 8, 8)
        x = self.conv2(x)          # -> (batch, 16, 8, 8)
        x = self.relu2(x)
        x = self.pool2(x)          # -> (batch, 16, 2, 2)
        x = self.flatten(x)        # -> (batch, 64)
        x = self.fc(x)             # -> (batch, 6)
        # Note: CrossEntropyLoss includes softmax, so we don't apply it here

        return x


# ==================== 3. Training Function ====================

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train a PyTorch model

    Arguments:
    model -- PyTorch model (nn.Module)
    train_loader -- DataLoader for training data
    val_loader -- DataLoader for validation data
    epochs -- number of training epochs
    lr -- learning rate
    device -- 'cpu' or 'cuda'

    Returns:
    history -- dictionary containing training history
    """
    model = model.to(device)

    # Define loss function and optimizer
    # For binary classification
    if model.__class__.__name__ == 'HappyModel':
        criterion = nn.BCELoss()  # Binary Cross Entropy
    else:
        criterion = nn.CrossEntropyLoss()  # For multiclass

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            if model.__class__.__name__ == 'HappyModel':
                loss = criterion(outputs, labels.float())
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels.argmax(dim=1))
                _, predicted = torch.max(outputs.data, 1)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_total += labels.size(0)
            if model.__class__.__name__ == 'HappyModel':
                train_correct += (predicted == labels).sum().item()
            else:
                train_correct += (predicted == labels.argmax(dim=1)).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if model.__class__.__name__ == 'HappyModel':
                    loss = criterion(outputs, labels.float())
                    predicted = (outputs > 0.5).float()
                else:
                    loss = criterion(outputs, labels.argmax(dim=1))
                    _, predicted = torch.max(outputs.data, 1)

                val_loss += loss.item()
                val_total += labels.size(0)
                if model.__class__.__name__ == 'HappyModel':
                    val_correct += (predicted == labels).sum().item()
                else:
                    val_correct += (predicted == labels.argmax(dim=1)).sum().item()

        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total

        history['loss'].append(epoch_train_loss)
        history['accuracy'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

    return history


# ==================== 4. Example Usage: Happy Model ====================

# Create model
happy_model = HappyModel()
print(happy_model)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.FloatTensor(Y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train the model
print("\nTraining Happy Model...")
history = train_model(happy_model, train_loader, test_loader, epochs=10, lr=0.001)

# ==================== 5. Example Usage: SIGNS Model ====================

# Load SIGNS dataset
X_train_signs_orig, Y_train_signs_orig, X_test_signs_orig, Y_test_signs_orig, classes = load_signs_dataset()

# Normalize and convert to one-hot
X_train_signs = X_train_signs_orig / 255.
X_test_signs = X_test_signs_orig / 255.
Y_train_signs = convert_to_one_hot(Y_train_signs_orig, 6).T
Y_test_signs = convert_to_one_hot(Y_test_signs_orig, 6).T

print("\nSIGNS Dataset Info:")
print("number of training examples = " + str(X_train_signs.shape[0]))
print("number of test examples = " + str(X_test_signs.shape[0]))
print("X_train shape: " + str(X_train_signs.shape))
print("Y_train shape: " + str(Y_train_signs.shape))

# Create model
conv_model = ConvolutionalModel(input_shape=(64, 64, 3), num_classes=6)
print(conv_model)

# Convert data to PyTorch tensors
X_train_signs_tensor = torch.FloatTensor(X_train_signs)
Y_train_signs_tensor = torch.FloatTensor(Y_train_signs)
X_test_signs_tensor = torch.FloatTensor(X_test_signs)
Y_test_signs_tensor = torch.FloatTensor(Y_test_signs)

# Create DataLoaders
train_signs_dataset = TensorDataset(X_train_signs_tensor, Y_train_signs_tensor)
test_signs_dataset = TensorDataset(X_test_signs_tensor, Y_test_signs_tensor)

train_signs_loader = DataLoader(train_signs_dataset, batch_size=64, shuffle=True)
test_signs_loader = DataLoader(test_signs_dataset, batch_size=64, shuffle=False)

# Train the model
print("\nTraining Convolutional Model for SIGNS...")
history_signs = train_model(conv_model, train_signs_loader, test_signs_loader, epochs=100, lr=0.001)

# ==================== 6. Visualization ====================

def plot_history(history, title='Model Training History'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Plot training history
plot_history(history_signs, title='SIGNS Model')

print("\n" + "="*50)
print("PyTorch CNN Application Complete!")
print("="*50)
print("\nKey Differences from Keras:")
print("1. PyTorch uses NCHW format (batch, channels, height, width)")
print("2. Keras uses NHWC format (batch, height, width, channels)")
print("3. We use .permute(0, 3, 1, 2) to convert between formats")
print("4. nn.Module subclassing instead of Sequential/Functional API")
print("5. Explicit training loops instead of model.fit()")
print("6. CrossEntropyLoss includes softmax, so no explicit softmax in forward()")
print("7. DataLoader for batching instead of tf.data.Dataset")

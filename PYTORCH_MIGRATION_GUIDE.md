# PyTorch Migration Guide

## Quick Start with PyTorch Version

### Installation

```bash
# Install PyTorch 2.5+ and dependencies
pip install -r requirements.txt

# Or install PyTorch directly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Major Differences from TensorFlow Version

## 1. Model Definition

### TensorFlow/Keras (Old)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    Dense(10, activation='softmax')
])
```

### PyTorch (New)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.fc(x)
        return x

model = MyModel()
```

## 2. Training Loop

### TensorFlow/Keras (Old)
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### PyTorch (New)
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

## 3. Data Format

### TensorFlow uses NHWC (Batch, Height, Width, Channels)
```python
# TensorFlow image shape
image_shape = (32, 224, 224, 3)  # 32 images, 224x224, RGB
```

### PyTorch uses NCHW (Batch, Channels, Height, Width)
```python
# PyTorch image shape
image_shape = (32, 3, 224, 224)  # 32 images, RGB, 224x224

# Convert if needed
tf_image = np.random.rand(1, 224, 224, 3)
pytorch_image = torch.from_numpy(tf_image.transpose(0, 3, 1, 2))
```

## 4. Common Operations

| Operation | TensorFlow/Keras | PyTorch |
|-----------|-----------------|---------|
| Matrix multiply | `tf.matmul(A, B)` | `torch.matmul(A, B)` or `A @ B` |
| ReLU | `tf.nn.relu(x)` | `F.relu(x)` or `nn.ReLU()(x)` |
| Softmax | `tf.nn.softmax(x)` | `F.softmax(x, dim=1)` |
| One-hot | `tf.keras.utils.to_categorical(y, C)` | `F.one_hot(y, num_classes=C)` |
| Reshape | `tf.reshape(x, shape)` | `x.reshape(shape)` or `x.view(shape)` |
| Transpose | `tf.transpose(x, [2, 0, 1])` | `x.permute(2, 0, 1)` |
| Concatenate | `tf.concat([a, b], axis=1)` | `torch.cat([a, b], dim=1)` |

## 5. Pre-trained Models

### TensorFlow/Keras (Old)
```python
from tensorflow.keras.applications import VGG16
model = VGG16(weights='imagenet')
```

### PyTorch (New)
```python
import torchvision.models as models
model = models.vgg16(pretrained=True)
# Or for newer PyTorch versions:
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
```

## 6. Saving and Loading Models

### TensorFlow/Keras (Old)
```python
model.save('my_model.h5')
model = tf.keras.models.load_model('my_model.h5')
```

### PyTorch (New)
```python
# Save
torch.save(model.state_dict(), 'my_model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('my_model.pth'))
model.eval()
```

## 7. Gradient Computation

### PyTorch requires explicit gradient control:
```python
# Disable gradients (for inference)
with torch.no_grad():
    output = model(input)

# Enable gradients (training mode)
model.train()

# Disable gradients (eval mode)
model.eval()
```

## 8. Data Loading

### PyTorch DataLoader
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 9. Moving to GPU

```python
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = model.to(device)

# Move data to GPU
inputs = inputs.to(device)
labels = labels.to(device)
```

## 10. Common Layers Mapping

| TensorFlow/Keras | PyTorch |
|-----------------|---------|
| `Dense(128)` | `nn.Linear(in_features, 128)` |
| `Conv2D(32, (3,3))` | `nn.Conv2d(in_channels, 32, kernel_size=3)` |
| `MaxPooling2D((2,2))` | `nn.MaxPool2d(kernel_size=2)` |
| `Flatten()` | `nn.Flatten()` or `x.view(x.size(0), -1)` |
| `Dropout(0.5)` | `nn.Dropout(0.5)` |
| `BatchNormalization()` | `nn.BatchNorm2d(num_features)` |
| `LSTM(128)` | `nn.LSTM(input_size, 128)` |
| `Embedding(vocab, 128)` | `nn.Embedding(vocab, 128)` |

## 11. Loss Functions

| Task | TensorFlow/Keras | PyTorch |
|------|-----------------|---------|
| Binary classification | `binary_crossentropy` | `nn.BCELoss()` or `nn.BCEWithLogitsLoss()` |
| Multi-class | `categorical_crossentropy` | `nn.CrossEntropyLoss()` |
| Sparse multi-class | `sparse_categorical_crossentropy` | `nn.CrossEntropyLoss()` |
| Regression | `mean_squared_error` | `nn.MSELoss()` |

## 12. Optimizers

| TensorFlow/Keras | PyTorch |
|-----------------|---------|
| `tf.keras.optimizers.Adam(lr=0.001)` | `torch.optim.Adam(params, lr=0.001)` |
| `tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)` | `torch.optim.SGD(params, lr=0.01, momentum=0.9)` |
| `tf.keras.optimizers.RMSprop(lr=0.001)` | `torch.optim.RMSprop(params, lr=0.001)` |

## 13. Batch Normalization

```python
# TensorFlow automatically switches between train/test mode
# PyTorch requires explicit control

# Training
model.train()  # Sets BatchNorm to training mode

# Inference
model.eval()   # Sets BatchNorm to eval mode
```

## 14. Custom Training Loop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {100*correct/total:.2f}%')
```

## 15. Debugging Tips

### Check tensor shapes
```python
print(x.shape)  # PyTorch
print(x.size()) # Also works in PyTorch
```

### Check for NaN
```python
assert not torch.isnan(x).any(), "Tensor contains NaN!"
```

### Gradient checking
```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

## Resources

- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **PyTorch Examples:** https://github.com/pytorch/examples
- **Torchvision Models:** https://pytorch.org/vision/stable/models.html

## Common Pitfalls

1. **Forgetting to call `.backward()`** - No gradients will be computed
2. **Forgetting `optimizer.zero_grad()`** - Gradients will accumulate
3. **Not calling `model.train()`/`model.eval()`** - BatchNorm/Dropout won't work correctly
4. **Wrong tensor shapes** - Remember NCHW format for images
5. **Not moving data to GPU** - Mixing CPU and GPU tensors causes errors
6. **Using wrong loss function** - CrossEntropyLoss expects class indices, not one-hot

---

*For specific course assignments, refer to converted notebooks and utility files for examples.*

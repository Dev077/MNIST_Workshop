# PyTorch Documentation - Quick Reference

## Installation
```bash
pip install torch torchvision torchaudio
```

## Tensors - Basics

### Creating Tensors
```python
import torch

# From data
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Initialization
torch.zeros(3, 4)           # All zeros
torch.ones(3, 4)            # All ones
torch.empty(3, 4)           # Uninitialized
torch.randn(3, 4)           # Normal distribution
torch.rand(3, 4)            # Uniform [0, 1)
torch.arange(0, 10, 2)      # Range [0, 10) step 2
torch.linspace(0, 10, 5)    # 5 evenly spaced values
torch.eye(3)                # Identity matrix
torch.full((3, 4), 7)       # Fill with value

# Like operations
x_like = torch.zeros_like(x)
x_like = torch.ones_like(x)
```

### Tensor Attributes
```python
x.shape                     # Size of tensor
x.dtype                     # Data type
x.device                    # cpu or cuda
x.requires_grad             # Track gradients
```

### Type Conversion
```python
x.float()                   # Convert to float32
x.double()                  # Convert to float64
x.long()                    # Convert to int64
x.int()                     # Convert to int32
x.to(torch.float64)         # Convert to specific type
x.to('cuda')                # Move to GPU
x.numpy()                   # Convert to numpy
torch.from_numpy(arr)       # Convert from numpy
```

## Tensor Operations

### Indexing & Slicing
```python
x[0]                        # First row
x[:, 1]                     # Second column
x[0:2, 1:3]                # Slice rows and columns
x[x > 0]                   # Boolean indexing
```

### Reshaping
```python
x.view(2, -1)              # Reshape (shares memory)
x.reshape(2, -1)           # Reshape (may copy)
x.squeeze()                # Remove dimensions of size 1
x.unsqueeze(0)             # Add dimension at position
x.permute(1, 0, 2)         # Permute dimensions
x.transpose(0, 1)          # Swap two dimensions
x.flatten()                # Flatten to 1D
```

### Mathematical Operations
```python
# Element-wise
x + y, x - y, x * y, x / y
x.add(y), x.sub(y), x.mul(y), x.div(y)
x ** 2, x.pow(2)           # Power
x.sqrt()                   # Square root
x.exp()                    # Exponential
x.log()                    # Natural log
torch.abs(x)               # Absolute value
torch.sin(x), torch.cos(x) # Trigonometric

# Matrix operations
x @ y                      # Matrix multiplication
torch.mm(x, y)            # Matrix multiplication
torch.matmul(x, y)        # Batch matrix multiplication
x.T                        # Transpose
torch.inverse(x)           # Matrix inverse

# Aggregation
x.sum()                    # Sum all elements
x.sum(dim=0)              # Sum along dimension
x.mean()                   # Mean
x.max()                    # Maximum value
x.max(dim=0)              # Max along dimension (returns value, index)
x.argmax()                 # Index of maximum
x.min(), x.argmin()       # Minimum operations
```

### Broadcasting
```python
# PyTorch automatically broadcasts tensors
x = torch.randn(3, 1)
y = torch.randn(1, 4)
z = x + y                  # Result is (3, 4)
```

## Autograd - Automatic Differentiation

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x

# Compute gradients
y.backward()               # Compute dy/dx
print(x.grad)              # Access gradient

# Disable gradient tracking
with torch.no_grad():
    y = x ** 2

# Detach from computation graph
y = x.detach()

# Zero gradients (important in training loops)
x.grad.zero_()
```

## Neural Networks

### Building Models
```python
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MyModel(784, 128, 10)
```

### Common Layers
```python
# Fully connected
nn.Linear(in_features, out_features)

# Convolutional
nn.Conv2d(in_channels, out_channels, kernel_size)
nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d(output_size=(1, 1))

# Normalization
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Dropout
nn.Dropout(p=0.5)

# Recurrent
nn.RNN(input_size, hidden_size, num_layers)
nn.LSTM(input_size, hidden_size, num_layers)
nn.GRU(input_size, hidden_size, num_layers)

# Activation (can also use functional versions)
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
```

### Sequential Models
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)
```

## Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()      # For multi-class
criterion = nn.BCELoss()                # Binary cross-entropy
criterion = nn.BCEWithLogitsLoss()      # BCE with sigmoid

# Regression
criterion = nn.MSELoss()                # Mean squared error
criterion = nn.L1Loss()                 # Mean absolute error
criterion = nn.SmoothL1Loss()           # Huber loss

# Usage
loss = criterion(predictions, targets)
```

## Optimizers

```python
import torch.optim as optim

# Common optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
```

## Training Loop

```python
# Training mode
model.train()

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move to device
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()

# Evaluation mode
model.eval()
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data)
        # Compute metrics
```

## Data Loading

```python
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader
dataset = MyDataset(data, labels)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # For GPU
)
```

## Saving & Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## GPU Usage

```python
# Check availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move to GPU
model = model.to(device)
x = x.to(device)

# Multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## Useful Functions

```python
# Parameter count
sum(p.numel() for p in model.parameters())

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers
for param in model.fc2.parameters():
    param.requires_grad = True

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
```

## TorchVision (Computer Vision)

```python
import torchvision
import torchvision.transforms as transforms

# Transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

# Pretrained models
model = torchvision.models.resnet50(pretrained=True)
```

"""
Training script for ResNet50 on MNIST dataset (grayscale images)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from resnet import ResNet50

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
num_epochs = 100
batch_size = 64
learning_rate = 0.001
num_classes = 10  # MNIST has 10 classes (digits 0-9)
img_channels = 1  # Grayscale images

# Data augmentation and normalization
# MNIST images are 28x28, we'll resize to 224x224 for ResNet50
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load MNIST dataset
print("Loading MNIST dataset...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform_train,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform_test,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f'Training samples: {len(train_dataset)}')
print(f'Test samples: {len(test_dataset)}')

# Initialize model
print("Initializing ResNet50 for grayscale images...")
model = ResNet50(img_channels=img_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print every 100 batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(test_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

# Training loop
print("\nStarting training...")
best_acc = 0.0
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)

    # Validate
    val_loss, val_acc = validate(model, test_loader, criterion)

    # Update learning rate
    scheduler.step()

    # Save history
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)

    epoch_time = time.time() - epoch_start

    print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'  Time: {epoch_time:.2f}s')
    print('-' * 60)

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'resnet50_mnist_best.pth')
        print(f'✓ Best model saved with accuracy: {best_acc:.2f}%')

total_time = time.time() - start_time
print(f'\nTraining completed in {total_time/60:.2f} minutes')
print(f'Best validation accuracy: {best_acc:.2f}%')

# Final evaluation
print("\nFinal evaluation on test set...")
test_loss, test_acc = validate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'img_channels': img_channels,
    'test_acc': test_acc,
    'training_history': training_history
}, 'resnet50_mnist_final.pth')
print('Final model saved as resnet50_mnist_final.pth')

# Per-digit accuracy
print("\nPer-digit accuracy:")
digit_correct = [0] * 10
digit_total = [0] * 10

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(labels.size(0)):
            label = labels[i]
            digit_correct[label] += (predicted[i] == label).item()
            digit_total[label] += 1

for i in range(10):
    acc = 100 * digit_correct[i] / digit_total[i] if digit_total[i] > 0 else 0
    print(f'Digit {i}: {acc:.2f}%')

print("\n✓ Training complete!")

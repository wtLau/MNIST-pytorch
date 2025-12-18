import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim



# why convert every input pixel to 0-1? what if we keep it in 0-255
# why normalize. what happens if we don't normalize


# Define a transform to convert the data to tensor and normalize it
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define a transform with data augmentation
transform_with_rotation = transforms.Compose([
    # RandomRotation seems rotates everything by about the same amount. wtf?
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training dataset with normalization
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# Load the training dataset with data augmentation
train_dataset_rotated = datasets.MNIST(root='data', train=True, download=True, transform=transform_with_rotation)

# Load the test dataset with normalization
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

def create_rotated_69_dataset(base):
    """Filter 6/9, rotate 180°, swap labels 6<->9"""
    result = []
    for img, label in base:
        if label in [6, 9]:
            img_rotated = transforms.functional.rotate(img, 180)
            result.append((img_rotated, 9 if label == 6 else 6))
    return result

def create_rotated_4_dataset(base):
    """Filter 4, rotate 180°, keep label"""
    result = []
    for img, label in base:
        if label == 4:
            img_rotated = transforms.functional.rotate(img, 180)
            result.append((img_rotated, label))
    return result

# Combine datasets
train_dataset = torch.utils.data.ConcatDataset([
    train_dataset,
    train_dataset_rotated,
    create_rotated_69_dataset(train_dataset),
    create_rotated_4_dataset(train_dataset)
])

# Define batch size
# why 64? and how does this influence training process?
batch_size = 64

# Create data loader for the training dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Create data loader for the test dataset
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# # Function to visualize images
# def show_images(images, labels):
#     fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
#     for img, label, ax in zip(images, labels, axes):
#         ax.imshow(img.squeeze(), cmap='gray')
#         ax.set_title(f'Label: {label}')
#         ax.axis('off')
#     plt.show()

# # Get a batch of images from the training data loader
# data_iter = iter(train_loader)
# images, labels = next(data_iter)

# # Display the first 5 images
# show_images(images[:5], labels[:5])

# what is feedforward network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # what is fc?
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

model = SimpleNN()

# Define loss function and optimizer
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print("Training complete!")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

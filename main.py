import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Credit
# -----------------------------
# https://mljourney.com/loading-the-mnist-dataset-in-pytorch-comprehensive-guide/

# -----------------------------
# 1. DATA PREPROCESSING
# -----------------------------

# A "transform" describes how we want to modify the images
# before feeding them into the model.

# For training:
# - RandomRotation(10): rotate slightly to make the model more robust
# - ToTensor(): convert image to PyTorch tensor
# - Normalize: scale pixel values from [0,1] → [-1,1] for easier training
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# For testing, we DO NOT apply rotation. Test data should remain untouched.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 2. LOADING THE DATASETS
# -----------------------------

# MNIST = dataset of handwritten digits (0–9)
# 'train=True' → training set
# 'train=False' → test set
# 'download=True' → download if missing
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

# -----------------------------
# 3. CREATING DATA LOADERS
# -----------------------------
# DataLoader groups the data into batches.
# A batch is a small chunk of the dataset processed at once (much faster).
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# shuffle=True → mix the training data every epoch

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 4. HELPER TO DISPLAY IMAGES
# -----------------------------

# Our normalize transformed images into [-1, 1].
# To display them properly, we reverse it.
def unnormalize(img):
    return img * 0.5 + 0.5

# Display a grid of images with labels (8x8 → 64 images)
def show_images(images, labels):
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    axes = axes.flatten()

    for img, label, ax in zip(images, labels, axes):
        ax.imshow(unnormalize(img).squeeze(), cmap='gray')  # squeeze removes extra channel dim
        ax.set_title(label.item())
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Grab 1 batch from the loader for visualization
data_iter = iter(train_loader)
images, labels = next(data_iter)
show_images(images[:64], labels[:64])

# -----------------------------
# 5. BUILDING A SIMPLE NEURAL NETWORK
# -----------------------------
# A very simple fully-connected (dense) network:
# Input: 28x28 image → flatten to 784 values
# Hidden layer: 128 neurons
# Output layer: 10 classes (digits 0–9)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # first layer: 784 → 128
        self.relu = nn.ReLU()               # activation function
        self.fc2 = nn.Linear(128, 10)       # second layer: 128 → 10
        self.softmax = nn.LogSoftmax(dim=1) # convert to log-probabilities

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Create the model
model = SimpleNN()

# -----------------------------
# 6. TRAINING SETUP
# -----------------------------
# Loss function: how "wrong" the model is
criterion = nn.NLLLoss()

# Optimizer: adjusts weights to reduce loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -----------------------------
# 7. TRAINING LOOP
# -----------------------------
epochs = 5

for epoch in range(epochs):
    running_loss = 0

    # Loop through all batches in training data
    for images, labels in train_loader:
        optimizer.zero_grad()           # reset gradients
        outputs = model(images)         # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # compute gradients
        optimizer.step()                # update weights
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

# -----------------------------
# 8. EVALUATION
# -----------------------------
# We check how well the model performs on unseen test data
correct = 0
total = 0

with torch.no_grad():  # no gradient calculation needed
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # pick highest-probability class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# ✅ 1. Save Your Model After Training
torch.save(model.state_dict(), "model.pth")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from conv_nn import CNN

# Load and preprocess images using NumPy
image_data = []
for file_path in ['data/cat.npy', 'data/dog.npy', 'data/airplane.npy', 'data/smiley face.npy']:
    gray_image = np.load(file_path)
    expanded_image = np.expand_dims(gray_image, axis=-1)
    rgb_image = np.repeat(expanded_image, 3, axis=-1)
    resized_image = np.resize(rgb_image, (28, 28, 3))
    image_data.append(resized_image)
preprocessed_images = np.array(image_data)
preprocessed_images = preprocessed_images.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

# Convert NumPy array to PyTorch tensor
preprocessed_images_tensor = torch.tensor(preprocessed_images).permute(0, 3, 1, 2)

# Define your data loader
labels = torch.tensor([0, 1, 2, 3])  # Example labels
dataset = TensorDataset(preprocessed_images_tensor, labels)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define the CNN model and move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(28, 28, num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), 'trained_model.pth')

print('Training finished. Saved as trained_model.pth ')
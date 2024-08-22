import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from time import time

# Define a custom transform to binarize pixel values
class BinarizeTransform:
    def __call__(self, x):
        return (x > 0.5).float()

# Compose the transformations: convert to grayscale, then binarize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    BinarizeTransform(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset with the custom grayscale and binarize transform
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Convert PyTorch DataLoader to numpy arrays
def convert_to_numpy(data_loader):
    data_list, label_list = [], []
    for images, labels in data_loader:
        data_list.append(images.numpy())
        label_list.append(labels.numpy())
    data_array = np.concatenate(data_list, axis=0)
    label_array = np.concatenate(label_list, axis=0)
    return data_array, label_array

X_train, Y_train = convert_to_numpy(train_loader)
X_test, Y_test = convert_to_numpy(test_loader)

# Reshape data to fit the Tsetlin Machine input requirements
X_train = X_train.reshape(-1, 32, 32)
X_test = X_test.reshape(-1, 32, 32)

# Binarize data (already done in the transform, this is just a safeguard)
X_train = np.where(X_train >= 0.5, 1, 0)
X_test = np.where(X_test >= 0.5, 1, 0)
print(X_train.shape)

# Initialize the Tsetlin Machine
tm = MultiClassConvolutionalTsetlinMachine2D(2000, 50 * 100, 5.0, (10, 10), weighted_clauses=True)

print("\nAccuracy over 30 epochs:\n")
for i in range(30):  # Adjust the number of epochs as needed
    start = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop = time()

    result = 100 * (tm.predict(X_test) == Y_test).mean()

    print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result, stop - start))

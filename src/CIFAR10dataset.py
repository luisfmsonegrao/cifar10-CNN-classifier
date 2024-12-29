import torch
from torchvision import datasets, transforms
from torchvision.datasets import vision

class CIFAR10dataset(vision.VisionDataset):
    def __init__(self, root, train=True, transform=None,target_transform=None):
        """
        Args:
            root (string): Directory where the dataset is stored.
            train (bool): If True, load the training data, otherwise load test data.
            transform (callable, optional): Optional transform to be applied to the samples.
        """
        self.root = root
        self.cifar10_data = datasets.CIFAR10(root=root, train=train, download=False)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar10_data)

    def __getitem__(self, idx):
        img, target = self.cifar10_data[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

""" # Example usage:
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing with mean and std
])

# Create the dataset
train_dataset = CIFAR10dataset(root='./data', train=True, transform=transform)
test_dataset = CIFAR10dataset(root='./data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# You can now use train_loader and test_loader in your training loop. """
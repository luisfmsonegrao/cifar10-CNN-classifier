import torch
from torchvision import datasets, transforms
from torchvision.datasets import vision

class CIFAR10dataset(vision.VisionDataset):
    def __init__(self, root, train=True, transform=None,target_transform=None):
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
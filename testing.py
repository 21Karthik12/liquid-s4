import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        if train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')
        self.transform = transform
        self.classes = [cls for cls in os.listdir(self.root_dir) if cls != '.ipynb_checkpoints']
        self.files = [os.path.join(self.root_dir, cls, file) for cls in self.classes for file in os.listdir(os.path.join(self.root_dir, cls))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path)
        label = self.classes.index(os.path.basename(os.path.dirname(img_path)))
        if self.transform:
            image = self.transform(image)
        return image, label

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Lambda(lambda x: x.view(3, 102400).t())
])
transform_train = transform_test = transform

trainset = CustomDataset(
    root_dir='./seed_data', train=True, transform=transform_train)
trainset, _ = split_train_val(trainset, val_split=0.1)

valset = CustomDataset(
    root_dir='./seed_data', train=True, transform=transform_train)
_, valset = split_train_val(valset, val_split=0.1)

testset = CustomDataset(
    root_dir='./seed_data', train=False, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=64, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=4)

print(trainloader.dataset.dataset)
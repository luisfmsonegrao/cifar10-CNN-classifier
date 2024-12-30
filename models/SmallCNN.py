import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.c1 = nn.Conv2d(3,8,3,stride=1,padding=1,bias=False)#8x32x32
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2,padding=0)#8x16x16
        self.c2 = nn.Conv2d(8,16,3,stride=1,padding=1,bias=False)#16x16x16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2,padding=0)#16x8x8
        self.c3 = nn.Conv2d(16,32,3,stride=1,padding=1,bias=False)#32x8x8
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2,padding=0)#32*4*4  
        self.lin1 = nn.Linear(512,64,bias=True)
        self.bn4 = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64,32,bias=True)
        self.bn5 = nn.BatchNorm1d(32)
        self.lin3 = nn.Linear(32,10,bias=True)

    def forward(self,x):
        out = self.dropout(x)
        out = self.pool1(self.relu(self.bn1(self.c1(out))))
        out = self.pool2(self.relu(self.bn2(self.c2(out))))
        out = self.pool3(self.relu(self.bn3(self.c3(out))))
        out = torch.flatten(out, start_dim=1)
        out = self.relu(self.bn4(self.lin1(out)))
        out = self.relu(self.bn5(self.lin2(out)))
        out = self.lin3(out)
        return out
        

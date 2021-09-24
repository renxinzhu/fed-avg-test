import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import generate_test_dataloader
from torch.utils.data.dataloader import DataLoader
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
'''

class MNISTConvNet(nn.Module):

    def __init__(self,dataloader: DataLoader):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dataloader = dataloader

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def validate(self):
        correct = 0
        dataloader = generate_test_dataloader(1000)
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.forward(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        return correct / len(dataloader.dataset)

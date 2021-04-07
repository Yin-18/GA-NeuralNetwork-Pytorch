import torch
import torch.nn as nn
import torch.nn.functional as F


# 0:用于MNIST 1:用于CIFAR-10
class LeNet_0(nn.Module):
    def __init__(self):
        super(LeNet_0, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(4*4*16, 10, bias=False)

    def forward(self, x):
        x = self.pool( F.relu(self.conv1(x) ) )
        x = self.pool( F.relu(self.conv2(x) ) )
        x = x.view(-1, 4*4*16)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=1)
        return x

    def show(self):
        print("conv1", self.conv1.weight.shape)
        print("conv2", self.conv2.weight.shape)
        print("fc1:", self.fc1.weight.shape)

class LeNet_1(nn.Module):
    def __init__(self):
        super(LeNet_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(4*4*16, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool( F.relu(self.conv1(x) ) )
        x = self.pool( F.relu(self.conv2(x) ) )
        x = x.view(-1, 4*4*16)
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LeNet_2(nn.Module):
    def __init__(self):
        super(LeNet_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool( F.relu(self.conv1(x) ) )
        x = self.pool( F.relu(self.conv2(x) ) )
        x = x.view(-1, 4*4*50)
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def show(self):
        print("conv1", self.conv1.weight.shape)
        print("conv2", self.conv2.weight.shape)
        print("fc1:", self.fc1.weight.shape)
        print("fc2:", self.fc2.weight.shape)






if __name__ == "__main__":
    inputs = torch.Tensor(6,1,28,28)
    net = LeNet_0()
    y=net(inputs)
    net.show()
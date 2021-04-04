import torch
import torch.nn as nn
import time

from models.lenet import LeNet_0, LeNet_1, LeNet_2
from scripts.data_loader import dataloader_MNIST
from algorithm.SGD import train_SGD


def main():
    save_root_path = './trainedModels/'

    learning_rate = 0.008
    epochs = 50
    train_momentum = 0.9
    batch_size = {"train": 16, "test": 64}

    train_loader, test_loader = dataloader_MNIST(sampler=False)
    net = LeNet_0()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=train_momentum)
    criterion = nn.CrossEntropyLoss()
    for i in range(epochs):
        print("epoch:%d------------------" % (i + 1))
        train_SGD(net, train_loader, optimizer, criterion, batch_size=batch_size["train"], epoch=i)
        if i % 10 == 9:
            localtime = time.asctime(time.localtime(time.time()))[4:].replace(' ', '_').replace(':', '_')
            temp_path = save_root_path + net._get_name() + '_' + localtime + '.pth'
            print("Saved the trained model! Path: " + temp_path)
            torch.save(net.state_dict(), temp_path)


if __name__ == "__main__":
    main()

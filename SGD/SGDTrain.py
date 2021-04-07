import torch
import torch.nn as nn
import time

from models.lenet import LeNet_0
from scripts.data_loader import dataloader_MNIST
from scripts.test_acc import test_Acc

def train_SGD(net, train_loader, optimizer, criterion, batch_size, epoch):
    net.train()

    runing_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # zero the parameter grddients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        runing_loss += loss.item()
        if batch_idx % 100==99:
            print("[%d, %5d] loss: %.5f" %
                    (epoch+1, (batch_idx+1)*batch_size, runing_loss/500))
            runing_loss = 0.0


def main():
    save_root_path = 'trainedModels/'

    learning_rate = 0.008
    epochs = 50
    train_momentum = 0.9
    batch_size = {"train": 32, "test": 32}

    train_loader, test_loader = dataloader_MNIST(data_path="../datasets",batch_size=batch_size, sampler=True, train_num_samples=10000, test_num_samples=2000)
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

    acc = test_Acc(net, test_loader)
    print("acc:", acc)


if __name__ == "__main__":
    main()

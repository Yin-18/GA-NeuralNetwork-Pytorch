import torch
from models.lenet import LeNet_0
from scripts.data_loader import dataloader_MNIST
from scripts.test_acc import test_Acc

def main():
    batch_size = {"train": 32, "test": 32}
    PATH = "trainedModels/LeNet_0_Apr__7_22_07_03_2021.pth"
    net = LeNet_0()
    net.load_state_dict(torch.load(PATH))
    train_loader, test_loader = dataloader_MNIST(data_path="../datasets",batch_size=batch_size, sampler=True, train_num_samples=10000, test_num_samples=2000)

    acc = test_Acc(net, test_loader)
    print(acc)


if __name__ ==  "__main__":
    main()
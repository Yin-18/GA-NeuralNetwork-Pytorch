import torch
from models.lenet import LeNet_0
from scripts.data_loader import dataloader_MNIST
from scripts.Test_Acc import test_Acc

def main():
    PATH = "trainedModels/LeNet_0Apr__3_16_43_25_2021.pth"
    net = LeNet_0()
    net.load_state_dict(torch.load(PATH))
    _ , test_loader = dataloader_MNIST(sampler=False)

    test_Acc(net, test_loader)


if __name__ ==  "__main__":
    main()
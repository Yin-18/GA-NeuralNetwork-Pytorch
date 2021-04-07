import torch
from torchvision import datasets, transforms


# 整个数据分布的很均匀 无需其他采样操作
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

def dataloader_MNIST(data_path="./datasets/",batch_size={"train":64, "test":64}, sampler=False,
                     train_num_samples=10000, test_num_samples=2000):
    assert train_num_samples<=60000 and test_num_samples<=10000, Exception("输入num值过大")
    train_set = datasets.MNIST(root=data_path, train=True, transform=data_transform, download=True)
    test_set = datasets.MNIST(root=data_path, train=False, transform=data_transform, download=True)

    if sampler==True:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size["train"],sampler=range(train_num_samples))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size["test"],sampler=range(test_num_samples))
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size["train"], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size["test"], shuffle=True)

    return (train_loader, test_loader)

if __name__ == "__main__":
    train_loader, test_loader = dataloader_MNIST(data_path="../datasets/",batch_size={"train":10000, "test":1000},train_num_samples=10000, test_num_samples=2000,sampler=True)
    sum = [0 for i in range(10)]
    for i,(x, y) in enumerate(train_loader):
        for j in range(10):
            sum[j] += (y==j).sum()
    print(sum)


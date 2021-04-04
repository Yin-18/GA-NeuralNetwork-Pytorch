import torch
from torchvision import datasets, transforms

data_path = "./datasets/"
sampler_weights = [1 for i in range(10)]

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])



train_set = datasets.MNIST(root=data_path, train=True, transform=data_transform, download=True)
test_set = datasets.MNIST(root=data_path, train=False, transform=data_transform, download=True)



def dataloader_MNIST(batch_size={"train":16, "test":64}, sampler=False, trainsetPerNum=2000):
    if sampler==True:
        Weight_Sampler = torch.utils.data.WeightedRandomSampler(sampler_weights, trainsetPerNum, replacement=True) #replacement=False会报个错 没解决
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size["train"],sampler=Weight_Sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size["train"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size["test"], shuffle=True)
    return (train_loader, test_loader)

if __name__ == "__main__":
    batch_size={"train":16, "test":64}
    print("sampler_weights: ", sampler_weights)
    train, test = dataloader_MNIST()
    print("len(train_loader)={}\nlen(test_loader)={}".format( len(train)*batch_size["train"] , len(test)*batch_size["test"] ))
    
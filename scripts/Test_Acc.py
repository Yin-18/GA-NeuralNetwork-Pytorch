import torch

def test_Acc(net, test_loader):
    '''
    返回准确率
    :param net: 网络
    :param test_loader: 测试集
    :return: 准确率
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print('Accuracy of the network: %.2f %%' % (100 * correct / total))
    return correct / total

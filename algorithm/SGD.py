import torch

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




def test_SGD(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (100 * correct / total))


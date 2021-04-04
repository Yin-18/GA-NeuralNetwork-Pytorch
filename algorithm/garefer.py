# -*- coding: utf-8 -*-

import numpy as np
import copy
import torch
import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms


BATCH_SIZE = 128

transform_train=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),
])

transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))
])

trainset=torchvision.datasets.CIFAR10(
        root='/Users/dengxq/Downloads',train=True,download=False,transform=transform_train)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

testset=torchvision.datasets.CIFAR10(
    root='/Users/dengxq/Downloads',train=False,download=False,transform=transform_test)
testloader=torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4)


class Net(torch.nn.Module):
    def __init__(self, layers=None, fc=None):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.05, inplace=False),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.05, inplace=False),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.05, inplace=False),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AvgPool2d(4, 4)
        )

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def set_layer(self, layers, fc):
        self.layers = layers
        self.fc = fc

class ElitismGA:
    def __init__(self, _pop_size, _r_mutation, _p_mutation,
                 _epochs, _elite_num, _mating_pool_size, _batch_size=32):
        # input params
        self.pop_size = _pop_size
        self.r_mutation = _r_mutation
        self.p_mutation = _p_mutation  # for generational
        self.epochs = _epochs
        self.elite_num = _elite_num  # for elitism
        self.mating_pool_size = _mating_pool_size  # for elitism
        self.batch_size = _batch_size
        # other params
        self.chroms = []
        self.evaluation_history = []
        self.stddev = 0.5
        self.criterion = nn.CrossEntropyLoss()
        self.model = None

    def initialization(self):
        for i in range(self.pop_size):
            net = Net()
            self.chroms.append(net)
        print('network initialization({}) finished.'.format(self.pop_size))

    def train(self):
        print('Elitism GA is training...')
        self.initialization()
        with torch.no_grad():
            for epoch in range(self.epochs):
                for step, (batch_x, batch_y) in enumerate(trainloader):
                    evaluation_result = self.evaluation(batch_x, batch_y, False)
                    self.selection(evaluation_result)

    def test(self):
        print('------ Test Start -----')
        correct = 0
        total = 0
        with torch.no_grad():
            for test_x, test_y in testloader:
                # images, labels = test_x.cuda(), test_y.cuda()
                images, labels = test_x, test_y
                output = self.model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy of the model is: %.4f %%' % accuracy)
        return accuracy

    def selection(self, evaluation_result):
        sorted_evaluation = sorted(evaluation_result, key=lambda x: x['train_acc'])
        elites = [e['pop'] for e in sorted_evaluation[-self.elite_num:]]
        print('Elites: {}'.format(elites))
        children = [self.chroms[i] for i in elites]
        mating_pool = np.array([self.roulette_wheel_selection(evaluation_result) for _ in range(self.mating_pool_size)])
        pairs = []
        print('mating_pool')
        print(mating_pool)
        while len(children) < self.pop_size:
            pair = [np.random.choice(mating_pool) for _ in range(2)]
            pairs.append(pair)
            children.append(self.crossover(pair))
        print('Pairs: {}'.format(pairs))
        print('Cross over finished.')

        self.replacement(children)
        for i in range(self.elite_num, self.pop_size):  # do not mutate elites
            if np.random.rand() < self.p_mutation:
                mutated_child = self.mutation(i)
                del self.chroms[i]
                self.chroms.insert(i, mutated_child)

    def crossover(self, _selected_pop):
        if _selected_pop[0] == _selected_pop[1]:
            return copy.deepcopy(self.chroms[_selected_pop[0]])

        chrom1 = copy.deepcopy(self.chroms[_selected_pop[0]])
        chrom2 = copy.deepcopy(self.chroms[_selected_pop[1]])

        chrom1_layers = list(chrom1.modules())
        chrom2_layers = list(chrom2.modules())

        child = torch.nn.Sequential()
        fc = None
        for i in range(len(chrom1_layers)):
            layer1 = chrom1_layers[i]
            layer2 = chrom2_layers[i]

            if isinstance(layer1, nn.Conv2d):
                child.add_module(str(i-2), layer1 if np.random.rand() < 0.5 else layer2)
            elif isinstance(layer1, nn.Linear):
                fc = layer1
            elif isinstance(layer1, (torch.nn.Sequential, Net)):
                pass
            else:
                child.add_module(str(i-2), layer1)

        chrom1.set_layer(child, fc)
        return chrom1

    def mutation(self, _selected_pop):
        child = torch.nn.Sequential()
        chrom = copy.deepcopy(self.chroms[_selected_pop])
        chrom_layers = list(chrom.modules())

        fc = None

        # 变异比例，选择几层进行变异
        for i, layer in enumerate(chrom_layers):
            if isinstance(layer, nn.Conv2d):
                if np.random.rand() < self.r_mutation:
                    weights = layer.weight.detach().numpy()
                    w = weights.astype(np.float32) + np.random.normal(0, self.stddev, weights.shape).astype(np.float32)

                    layer.weight = torch.nn.Parameter(torch.from_numpy(w))
                child.add_module(str(i-2), layer)
            elif isinstance(layer, nn.Linear):
                fc = layer
            elif isinstance(layer, (torch.nn.Sequential, Net)):
                pass
            else:
                child.add_module(str(i-2), layer)
        print('Mutation({}) finished.'.format(_selected_pop))

        chrom.set_layer(child, fc)
        return chrom

    def replacement(self, _child):
        self.chroms[:] = _child
        print('Replacement finished.')

    def evaluation(self, batch_x, batch_y, _is_batch=True):
        cur_evaluation = []
        for i in range(self.pop_size):
            model = self.chroms[i]
            output = model(batch_x)
            train_loss = self.criterion(output, batch_y).item()

            _, predicted = torch.max(output.data, 1)
            total = batch_y.size(0)
            correct = (predicted == batch_y.data).sum().item()
            train_acc = 100 * correct / total

            cur_evaluation.append({
                'pop': i,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4),
            })
        best_fit = sorted(cur_evaluation, key=lambda x: x['train_acc'])[-1]
        self.evaluation_history.append({
            'iter': len(self.evaluation_history) + 1,
            'best_fit': best_fit,
            'avg_fitness': np.mean([e['train_acc'] for e in cur_evaluation]).round(4),
            'evaluation': cur_evaluation,
        })
        print('\nIter: {}'.format(self.evaluation_history[-1]['iter']))
        print('Best_fit: {}, avg_fitness: {:.4f}'.format(self.evaluation_history[-1]['best_fit'],
                                                         self.evaluation_history[-1]['avg_fitness']))
        self.model = self.chroms[best_fit['pop']]
        return cur_evaluation

    def roulette_wheel_selection(self, evaluation_result):
        sorted_evaluation = sorted(evaluation_result, key=lambda x: x['train_acc'])
        cum_acc = np.array([e['train_acc'] for e in sorted_evaluation]).cumsum()
        extra_evaluation = [{'pop': e['pop'], 'train_acc': e['train_acc'], 'cum_acc': acc}
                            for e, acc in zip(sorted_evaluation, cum_acc)]
        rand = np.random.rand() * cum_acc[-1]
        for e in extra_evaluation:
            if rand < e['cum_acc']:
                return e['pop']
        return extra_evaluation[-1]['pop']


if __name__ == '__main__':
    g = ElitismGA(
        _pop_size=100,
        _p_mutation=0.1,
        _r_mutation=0.1,
        _epochs=20,
        _elite_num=20,
        _mating_pool_size=40,
        _batch_size=32
    )

    g.train()

    g.test()

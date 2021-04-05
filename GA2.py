import torch
import torch.nn as nn
import numpy as np
import copy
import pickle

from models.lenet import LeNet_0
from scripts.Test_Acc import test_Acc
from scripts.data_loader import dataloader_MNIST

# (6, 1, 5, 1)->150
# (16, 6, 5, 5)->2400
# (10, 256)->2560
# chrom [x0, x1, x2, x3, x4, x5...] * 5110

g_num_parameters = 5110

class Chrom():
    def __init__(self):
        self.parameters = None
        self.matation_strength = 2.0
        self.mean_range = (-1, 1)
        self.std_range = (0, 1)
        self.fitness = 0.0

        # 初始化染色体参数
        self._initialize()

    def _initialize(self):
        mean = np.random.random_sample() * (self.mean_range[1] - self.mean_range[0]) + self.mean_range[0]
        std = np.random.random_sample() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]

        # parameters是ndarray对象
        self.parameters = np.random.normal(mean, std, size=(g_num_parameters))


class GA_2_NeuralNet():
    def __init__(self, test_loader, pop_size=30, generations=1000, x_prob=0.8, m_prob=0.2):
        self.test_loader = test_loader
        self.x_prob = x_prob
        self.m_prob = m_prob
        self.pop_size = pop_size
        self.generations = generations
        self.best = []
        self.pop = []

    def initialize(self):  #初始化种群
        for i in range(self.pop_size):
            self.pop.append(Chrom())

        # 评价初始化的种群
        self._evaluate(self.pop)

    def evolution(self):
        gen0list = sorted(self.pop, key=lambda chrom: chrom.fitness, reverse=True)
        self.best.append(gen0list[0].fitness)
        print("[Gen 0]Best Chrom Fitness:%.4f" % (gen0list[0].fitness))
        # 开始进化
        for i in range(self.generations):
            self._evolution_step()  # 扩充了self.pop(由父代和子代组成 = 2*pop_size）
            self._environment_selection()   # 删减了self.pop(回到 pop_size)
            self.best.append(self.pop[0].fitness)
            print("[Gen %d]Best Chrom Fitness:%.4f" % (i + 1, self.pop[0].fitness))

            #保存进化中的种群
            if i % 5 == 4:
                self.save_population()

    def _evolution_step(self):
        # shuffle种群
        np.random.shuffle(self.pop)
        # 产生后代的列表
        offspring_list = []
        for _ in range(int(self.pop_size / 2)):
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            # crossover
            off1, off2 = self._crossover(parent1, parent2)
            # mutation
            off1 = self._mutation(off1)
            off2 = self._mutation(off2)
            # add to offspringlist
            offspring_list.append(off1)
            offspring_list.append(off2)

        self._evaluate(offspring_list)
        self.pop.extend(offspring_list)

    def _environment_selection(self):
        assert (len(self.pop) == 2 * self.pop_size)
        elites_rate = 0.2
        num_elites = int(elites_rate * len(self.pop))
        chrom_list = self.pop
        chrom_list.sort(key=lambda chrom: chrom.fitness, reverse=True)
        elites_list = chrom_list[0: num_elites]
        left_list = chrom_list[num_elites:]
        np.random.shuffle(left_list)

        for _ in range(self.pop_size - num_elites):
            chrom1_idx = np.random.randint(0, len(left_list))
            chrom2_idx = np.random.randint(0, len(left_list))
            if left_list[chrom1_idx].fitness > left_list[chrom2_idx].fitness:
                elites_list.append(left_list[chrom1_idx])
            else:
                elites_list.append(left_list[chrom2_idx])

        self.pop = elites_list

    def _evaluate(self, population: list):
        for chrom in population:
            net = self.parse_chrom(chrom)
            fitness = test_Acc(net, self.test_loader)
            chrom.fitness = fitness

    def _crossover(self, chrom1, chrom2):
        # 基于python传参可变类型的限制 需要进行deepcopy
        p1 = copy.deepcopy(chrom1)
        p2 = copy.deepcopy(chrom2)
        prob = np.random.random_sample(size=g_num_parameters)
        mask = (prob < self.x_prob)
        # Numpy布尔索引 交叉操作
        p1.parameters[mask], p2.parameters[mask] = p2.parameters[mask], p1.parameters[mask]
        return p1, p2

    def _mutation(self, chrom):
        # 这里不确定是不是对chrom进行了原地操作
        prob = np.random.random_sample(size=g_num_parameters)
        mask = (prob < self.m_prob)
        size = mask.sum()
        chrom.parameters[mask] = chrom.parameters[mask] + np.random.normal(0, chrom.matation_strength, size=size)
        return chrom

    def parse_chrom(self, chrom:Chrom):  # 解码染色体为神经网络
        conv1 = chrom.parameters[:150].reshape((6, 1, 5, 5), order='C')
        conv2 = chrom.parameters[150:2550].reshape((16, 6, 5, 5), order='C')
        fc1 = chrom.parameters[2550:].reshape((10, 256), order='C')

        net = LeNet_0()

        # conv1
        conv1_weight = torch.from_numpy(conv1).type(torch.FloatTensor)
        net.conv1.weight = nn.Parameter(conv1_weight)
        # conv2
        conv2_weight = torch.from_numpy(conv2).type(torch.FloatTensor)
        net.conv2.weight = nn.Parameter(conv2_weight)
        # fc1
        fc1_weight = torch.from_numpy(fc1).type(torch.FloatTensor)
        net.fc1.weight = nn.Parameter(fc1_weight)
        return net




    def _tournament_selection(self):  # 锦标赛选择算子
        chrom1_idx = np.random.randint(0, self.pop_size)
        chrom2_idx = np.random.randint(0, self.pop_size)
        if self.pop[chrom1_idx].fitness > self.pop[chrom2_idx].fitness:
            winner = self.pop[chrom1_idx]
        else:
            winner = self.pop[chrom2_idx]
        return winner

    def _rouletteWheelSelection(self):  # 轮盘赌选择算子
        pass

    def save_population(self):  # 保存种群
        acc = str(self.best[-1]).replace(".", "_")
        PATH = "./trainedGA2/" + "GA_bestAcc(" + acc + ")_size" + str(self.pop_size) + ".pkl"
        with open(PATH, "wb") as f:
            data = {"pop":self.pop, "best":self.best}
            try:
                pickle.dump(data, f)
                print("Saved the trained model! Path: " + PATH)
            except:
                print("Save Error!")

    def load_population(self, PATH):  # 加载种群
        with open(PATH, "rb") as f:
            data = pickle.load(f)

            self.pop = data["pop"]
            self.best = data["best"]




if __name__ == "__main__":
    _, test_loader = dataloader_MNIST(sampler=False)
    ga = GA_2_NeuralNet(test_loader)
    ga.initialize()
    ga.evolution()



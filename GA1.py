import torch
import torch.nn as nn
import numpy as np
import copy
import pickle

from models.lenet import LeNet_0
from scripts.Test_Acc import test_Acc
from scripts.data_loader import dataloader_MNIST


class Chrom():
    def __init__(self):
        self.parameters = {}
        self.mutation_strength = {}
        self.mean_range = (-1, 1)
        self.std_range = (0, 1)
        self.fitness = 0.0

        self._initialize()

    def __getitem__(self, item):
        return self.parameters[item]

    def _initialize(self):
        '''对self.parameters进行初始化'''
        layers_list = {"conv1": (6, 1, 5, 5), "conv2": (16, 6, 5, 5), "fc1": (10, 256)}

        for layer in layers_list:
            mean = np.random.random_sample() * (self.mean_range[1] - self.mean_range[0]) + self.mean_range[0]
            std = np.random.random_sample() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]
            self.parameters[layer] = np.random.normal(mean, std, size=layers_list[layer])


class GA_NeuralNet():
    def __init__(self, test_loader, pop_size=20, generations=1000, x_prob=0.8, m_prob=0.2):
        self.test_loader = test_loader
        self.x_prob = x_prob
        self.m_prob = m_prob
        self.pop_size = pop_size
        self.generations = generations
        self.best = []
        self.pop = []

    def initialize(self):
        """
        初始化一个种群
        """
        for i in range(self.pop_size):
            self.pop.append(Chrom())

        #评价初始化的种群
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
            if i % 10 == 9:
                self.save_population()

    def _evolution_step(self):
        np.random.shuffle(self.pop)
        offspring_list = []
        for _ in range(int(self.pop_size / 2)):
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            # crossover
            off1, off2 = self._crossover(parent1, parent2)
            # mutation
            self._mutation(off1)
            self._mutation(off2)
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
        p1 = copy.deepcopy(chrom1)
        p2 = copy.deepcopy(chrom2)
        if np.random.random_sample() < self.x_prob:
            p1.parameters["conv1"], p2.parameters["conv1"] = p2.parameters["conv1"], p1.parameters["conv1"]
        if np.random.random_sample() < self.x_prob:
            p1.parameters["conv2"], p2.parameters["conv2"] = p2.parameters["conv2"], p1.parameters["conv2"]
        if np.random.random_sample() < self.x_prob:
            p1.parameters["fc1"], p2.parameters["fc1"] = p2.parameters["fc1"], p1.parameters["fc1"]
        return p1, p2

    def _mutation(self, chrom):
        if np.random.random_sample() < self.m_prob:
            weight = chrom.parameters["conv1"]
            chrom.parameters["conv1"] = weight + np.random.normal(0, 1, size=weight.shape)
        if np.random.random_sample() < self.m_prob:
            weight = chrom.parameters["conv2"]
            chrom.parameters["conv2"] = weight + np.random.normal(0, 1, size=weight.shape)
        if np.random.random_sample() < self.m_prob:
            weight = chrom.parameters["fc1"]
            chrom.parameters["fc1"] = weight + np.random.normal(0, 1, size=weight.shape)



    def parse_chrom(self, chrom):
        net = LeNet_0()
        # conv1
        conv1_weight = torch.from_numpy(chrom.parameters["conv1"], ).type(torch.FloatTensor)
        net.conv1.weight = nn.Parameter(conv1_weight)
        # conv2
        conv2_weight = torch.from_numpy(chrom.parameters["conv2"]).type(torch.FloatTensor)
        net.conv2.weight = nn.Parameter(conv2_weight)
        # conv3
        fc1_weight = torch.from_numpy(chrom.parameters["fc1"]).type(torch.FloatTensor)
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

    def save_population(self):
        acc = str(self.best[-1]).replace(".", "_")
        PATH = "./trainedPopulation/" + "GA_bestAcc(" + acc + ")_size" + str(self.pop_size) + ".pkl"
        with open(PATH, "wb") as f:
            data = {"pop":self.pop, "best":self.best}
            try:
                pickle.dump(data, f)
                print("Saved the trained model! Path: " + PATH)
            except:
                print("Save Error!")

    def load_population(self, PATH):
        with open(PATH, "rb") as f:
            data = pickle.load(f)

            self.pop = data["pop"]
            self.best = data["best"]


if __name__ == "__main__":
    _, test_loader = dataloader_MNIST(sampler=False)
    ga = GA_NeuralNet(test_loader)
    ga.initialize()
    ga.evolution()

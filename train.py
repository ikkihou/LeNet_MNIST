# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 20:08
# @Author  : YIHUI-BAO
# @File    : train.py
# @Software: PyCharm
# @mail    : paulbao@mail.ecust.edu.cn


import matplotlib

matplotlib.use("Agg")

from module.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import torch
import time

TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT


def parse_arg():
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, required=True,
                    help="path to output loss/accuracy plot")
    ap.add_argument("-lr", default=0.001, type=float, help="learning rate")
    ap.add_argument("-bs", default=64, type=int, help="batch size")
    ap.add_argument("-epochs", default=10, type=int, help="the iteration times")

    args = ap.parse_args()

    return args


class Solver(object):
    def __init__(self, config):
        self.learning_rate = config.lr
        self.BatchSize = config.bs
        self.epochs = config.epochs
        self.model_output_path = config.model
        self.plot_output_path = config.plot
        self.model = None
        self.device = None
        self.optimizer = None
        self.criterion = None

    def load_data(self):
        print(f"[INFO] Loading the MNIST dataset ... ")
        self.trainData = KMNIST(root="./data", train=True, download=True, transform=ToTensor())
        self.testData = KMNIST(root="./data", train=False, download=True, transform=ToTensor())

        print(f"[INFO] generating the train/validation split ... ")
        self.numTrainSamples = int(len(self.trainData) * TRAIN_SPLIT)
        self.numValSamples = int(len(self.trainData) * VAL_SPLIT)

        (self.trainData, self.ValData) = random_split(self.trainData, [self.numTrainSamples, self.numValSamples],
                                                      generator=torch.Generator().manual_seed(42))

        self.TrainDataLoader = DataLoader(self.trainData, batch_size=self.BatchSize, shuffle=True)
        self.ValDataLoader = DataLoader(self.ValData, batch_size=self.BatchSize)
        self.TesTDataLoader = DataLoader(self.testData, batch_size=self.BatchSize)

        self.trainSteps = len(self.TrainDataLoader.dataset) // self.BatchSize
        self.valSteps = len(self.ValDataLoader.dataset) // self.BatchSize

    def load_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using {self.device} ... ")
        print(f"[INFO] Loading the LeNet Model ... ")
        self.model = LeNet(input_channels=1, num_classes=len(self.trainData.dataset.classes))
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.NLLLoss()
        self.H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train(self):
        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()

            totalTrainLoss = 0.0
            totalValLoss = 0.0

            trainCorrect = 0
            valCorrect = 0

            loop = tqdm(self.TrainDataLoader)
            for (inputs, labels) in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                pred = self.model(inputs)
                loss = self.criterion(pred, labels)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loop.set_description(f"Epoch[{epoch + 1}/{self.epochs}]")

                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == labels).type(torch.float).sum().item()

            with torch.no_grad():
                self.model.eval()

                for (inputs, labels) in self.ValDataLoader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    pred = self.model(inputs)
                    totalValLoss += self.criterion(pred, labels)
                    valCorrect += (pred.argmax(1) == labels).type(torch.float).sum().item()

            avgTrainLoss = totalTrainLoss / self.trainSteps
            avgValLoss = totalValLoss / self.valSteps

            trainCorrect = trainCorrect / len(self.TrainDataLoader.dataset)
            valCorrect = valCorrect / len(self.ValDataLoader.dataset)

            self.H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            self.H["train_acc"].append(trainCorrect)
            self.H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            self.H["val_acc"].append(valCorrect)

            print(f"#{'-' * 50}#")
            print(f"[INFO] EPOCH {epoch + 1}/{self.epochs} ")
            print(f"Train loss: {avgTrainLoss:.4f} | Train acc: {trainCorrect:.4f}")
            print(f"Val loss: {avgValLoss:.4f} | Val acc: {valCorrect:.4f}")
            print(f"#{'-' * 50}#")
            print(f"\n")

        end_time = time.time()
        print(f"[INFO] Training time: {end_time - start_time:.4f} seconds")

    def test(self):
        print(f"[INFO] Evaluating the network ... ")
        self.model.eval()

        pred = []
        labels = []

        for (inputs, label) in self.TesTDataLoader:
            inputs = inputs.to(self.device)
            pred.extend(self.model(inputs).argmax(1).cpu().detach().numpy())
            labels.extend(label.numpy())

        print(classification_report(labels, pred, target_names=self.trainData.dataset.classes))

    def plot(self):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), self.H["train_loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), self.H["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), self.H["train_acc"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), self.H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(self.plot_output_path)

    def save(self):
        print(f"[INFO] Saving the model ... ")
        torch.save(self.model.state_dict(), self.model_output_path)

    def run(self):
        self.load_data()
        self.load_model()
        self.train()
        self.test()
        self.plot()
        self.save()


if __name__ == '__main__':
    ap = parse_arg()
    solver = Solver(ap)
    solver.run()

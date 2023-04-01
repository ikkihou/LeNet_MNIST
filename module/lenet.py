# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 19:56
# @Author  : YIHUI-BAO
# @File    : lenet.py
# @Software: PyCharm
# @mail    : paulbao@mail.ecust.edu.cn


import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(LeNet, self).__init__()

        self.Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.Layer3 = nn.Sequential(
            nn.Linear(in_features=800, out_features=500),
            nn.ReLU(),
        )

        self.flatten = nn.Sequential(
            nn.Flatten()
        )

        self.Layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.flatten(x)
        x = self.Layer3(x)
        output = self.Layer4(x)

        return output

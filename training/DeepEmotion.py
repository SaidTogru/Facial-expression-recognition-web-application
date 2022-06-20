import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepEmotion(nn.Module):
    def __init__(self):
        super(DeepEmotion, self).__init__()
        self.ConvolutionLayer_1 = nn.Conv2d(1, 10, 3)
        self.ConvolutionLayer_2 = nn.Conv2d(10, 10, 3)
        self.ConvolutionLayer_3 = nn.Conv2d(10, 10, 3)
        self.ConvolutionLayer_4 = nn.Conv2d(10, 10, 3)
        self.MaxPooling_1 = nn.MaxPool2d(2, 2)
        self.MaxPooling_2 = nn.MaxPool2d(2, 2)
        self.BatchNormalization = nn.BatchNorm2d(10)
        self.FullyConnectedLayer_1 = nn.Linear(810, 50)
        self.FullyConnectedLayer_2 = nn.Linear(50, 7)
        self.FullyConnectedLayer_3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.FullyConnectedLayer_4 = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.FullyConnectedLayer_4[2].weight.data.zero_()
        self.FullyConnectedLayer_4[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def LocalizationNetwork(self, input):
        output = self.FullyConnectedLayer_3(input)
        output = output.view(-1, 640)
        theta = self.FullyConnectedLayer_4(output)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input.size())
        transformed = F.grid_sample(input, grid)
        return transformed

    def forward(self, input):
        output = self.LocalizationNetwork(input)
        output = input
        output = F.relu(self.ConvolutionLayer_1(output))
        output = self.ConvolutionLayer_2(output)
        output = F.relu(self.MaxPooling_1(output))

        output = F.relu(self.ConvolutionLayer_3(output))
        output = self.BatchNormalization(self.ConvolutionLayer_4(output))
        output = F.relu(self.MaxPooling_2(output))

        output = F.dropout(output)
        output = output.view(-1, 810)
        output = F.relu(self.FullyConnectedLayer_1(output))
        output = self.FullyConnectedLayer_2(output)

        return output

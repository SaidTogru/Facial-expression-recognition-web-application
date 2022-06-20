from torch import nn


def convolutional_block(inp, out, pool=False):
    layers = [nn.Conv2d(inp, out, kernel_size=3),
              nn.BatchNorm2d(out),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.cb1 = convolutional_block(1, 32, pool=False)
        self.cb2 = convolutional_block(32, 64, pool=True)
        self.cb3 = convolutional_block(64, 128, pool=True)
        self.classifier = nn.Sequential(nn.MaxPool2d(2, 2),
                                        nn.Flatten(),
                                        nn.Linear(128*31*31, 256),
                                        nn.Linear(256, 7))

    def forward(self, x):
        x = self.cb1(x)
        x = nn.Dropout(0.5)(x)
        x = self.cb2(x)
        x = nn.Dropout(0.5)(x)
        x = self.cb3(x)
        x = self.classifier(x)
        return x

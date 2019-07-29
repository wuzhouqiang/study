import torch
from torch import nn
import torch.nn.functional as F


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        return out + self.extra(x)


class ResNet(nn.Module):

    def __init__(self):     # b*1*28*28
        super(ResNet, self).__init__()

        self.layout1 = nn.Sequential(  # 1*28*28
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),  # 16*28*28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 1),  # 32*28*28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 64, 14, 14
        )

        self.blk1 = ResBlk(64, 128)
        self.blk2 = ResBlk(128, 256)
        self.blk3 = ResBlk(256, 512)

        self.out = nn.Sequential(
            nn.Linear(512 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.layout1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = x.view(x.size(0), -1)  # b , 512 * 14 * 14
        x = self.out(x)

        return x


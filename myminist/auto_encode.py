import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import visdom

# 定义超参数

batch_size = 128
learning_rate = 1e-3
epochs = 20
weight_decay = 1e-4


normTransform = transforms.Normalize((0.1307,), (0.3081,))

trainTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=trainTransform, download=False)

test_dataset = datasets.MNIST(root='./data', train=False, transform=trainTransform, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )

        self.decode = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 784)
        )

    def forward(self, x):

        batch_size = x.size(0)  # [1, 28, 28]
        x = x.view(batch_size, -1)
        x = self.encoder(x)
        x = self.decode(x)
        x = x.view(1, 28, 28)
        return x



def main():
    x, _ = iter(train_dataset).__next__()

    vis = visdom.Visdom()

    device = torch.device('cuda')
    modle = AE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(modle.parameters(), lr=learning_rate)
    loss = None
    for epoch in range(epochs):
        for step, (x, _) in enumerate(train_dataset):
            x = x.to(device)
            x_pred = modle(x)

            loss = criteon(x_pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch:', epoch, ' loss:', loss.item())

        x = iter(test_dataset).__next__()

        with torch.no_grad():
            x_hat = modle(x)

        vis.images(x, nrow=8, win='x', opts=dict(title='x'))

        vis.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()


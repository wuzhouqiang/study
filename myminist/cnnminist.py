import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
import visdom

# 定义超参数
batch_size = 128
learning_rate = 1e-3
num_epoches = 300
weight_decay = 1e-4

# 下载训练集 MNIST 手写数字训练集

# 数据预处理设置
# normMean = [0.4948052]
# normStd = [0.24580306]
normTransform = transforms.Normalize((0.1307,), (0.3081,))

trainTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=trainTransform, download=False)

test_dataset = datasets.MNIST(root='./data', train=False, transform=trainTransform, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

vis = visdom.Visdom()


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.layout1 = nn.Sequential(  # 1*28*28

            nn.Conv2d(1, 16, 3, padding=1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),         # 16*28*28
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 5, padding=2),  # 16*28*28
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.Conv2d(64, 128, 1),                   # 32 * 28*28
            nn.Dropout(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),    # 128, 14, 14
        )

        self.layout2 = nn.Sequential(
            nn.Linear(128 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layout1(x)
        # x = x.view(-1,32 * 14 * 14)
        x = x.view(x.size(0), -1)
        out = self.layout2(x)
        return out


model = CNNNet()
# model = RNN_net.RNN_net()
# model = res_net.ResNet()
# model.load_state_dict(torch.load('cnn_minist.pth'))

use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
device = torch.device('cpu')
if use_gpu:
    device = torch.device('cuda')
if use_gpu:
    model = model.cuda()

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# 开始训练
def train_net():
    global_step = 1

    for epoch in range(num_epoches):
        print('*' * 80)
        print('epoch {}'.format(epoch + 1))
        # scheduler.step()
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            # 向前传播
            out = model(img).to(device)
            loss = criterion(out, label).to(device)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            num_correct = (pred == label).sum()
            running_acc += num_correct.item()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('[{}/{}] Loss: {:.11f}, Acc: {:.11f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))

                vis.line([loss.item()], [global_step], win='loss', opts=dict(title='loss'), update='append')
                global_step += 50

        print('{} epoch, Loss: {:.11f}, Acc: {:.11f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))


        # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        # viz.line([loss.item()], [i], win='train_loss', update='append')

        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for data in test_loader:
            img, label = data
            # img = img.view(img.size(0), -1)
            if use_gpu:
                img =img.cuda()
                label =label.cuda()

            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.11f}, Acc: {:.11f}'.format(eval_loss / (len(
            test_dataset)), eval_acc / (len(test_dataset))))

        vis.line([eval_acc / len(test_dataset)], [global_step], win='Test Loss', opts=dict(title='Test Loss'), update='append')

        print('Time:{:.1f} s'.format(time.time() - since))


    # 保存模型
    torch.save(model.state_dict(), './cnn_minist.pth')


if __name__ == '__main__':
    train_net()

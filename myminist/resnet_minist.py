import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time


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


use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
device = torch.device('cpu')
if use_gpu:
    device = torch.device('cuda')


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)





# 定义loss和optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 开始训练
def train_net():


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

                # vis.line([loss.item()], [global_step], win='loss', opts=dict(title='loss'), update='append')
                # global_step += 50

        print('{} epoch, Loss: {:.11f}, Acc: {:.11f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))


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

        print('Time:{:.1f} s'.format(time.time() - since))


    # 保存模型
    torch.save(model.state_dict(), './cnn_minist.pth')


if __name__ == '__main__':
    # train_net()
    pass
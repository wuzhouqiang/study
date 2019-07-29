import torch
import torch.nn as nn


class RNN_net(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layer=2):
        super(RNN_net, self).__init__()
        self.rnn = nn.LSTM(input_size=in_feature, hidden_size=hidden_feature, num_layers=num_layer)
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self, x):  # b*1*28*28
        x = x.squeeze()
        x = x.permute(2, 0, 1)  # 28*b*28
        out, _ = self.rnn(x)     # out 28*b*100
        out = out[-1, :, :]      # b *100

        out = self.classifier(out)  #
        return out



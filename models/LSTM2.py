import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=70,
            batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(70)

        self.lstm2 = nn.LSTM(
            input_size=70,
            hidden_size=80,
            batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.lstm3 = nn.LSTM(
            input_size=80,
            hidden_size=70,
            batch_first=True)
        self.dropout3 = nn.Dropout(0.1)
        self.batchnorm3 = nn.BatchNorm1d(70)

        self.lstm4 = nn.LSTM(
            input_size=70,
            hidden_size=80,
            batch_first=True)
        self.dropout4 = nn.Dropout(0.2)
        self.batchnorm4 = nn.BatchNorm1d(80)

        self.fc = nn.Linear(80, configs.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, packed_input):
        # 解包PackedSequence
        padded_input, input_lengths = pad_packed_sequence(packed_input, batch_first=True)

        batch_size, max_num_window, window_size, _ = padded_input.size()

        # 重塑x以适应第一层LSTM的输入形状
        x = padded_input.reshape(batch_size * max_num_window, window_size, -1)

        # 通过第一层LSTM
        out, _ = self.lstm1(x)  # out: tensor of shape (窗口数(batch_size * max_num_window), 序列长度, 隐层数(维度))
        out = self.dropout1(out)

        out = out.reshape(-1, 70)  # 将形状调整为 (窗口数 * 窗口长度, hidden_size)
        out = self.batchnorm1(out)
        out = out.reshape(batch_size * max_num_window, window_size, 70)  # 将形状调整回 (窗口数, 窗口长度, hidden_size)

        # 通过第二层LSTM
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out = out.reshape(-1, 80)  # 将形状调整为 (窗口数 * 窗口长度, hidden_size)
        out = self.batchnorm2(out)
        out = out.reshape(batch_size * max_num_window, window_size, 80)  # 将形状调整回 (窗口数, 窗口长度, hidden_size)

        # 通过第三层LSTM
        _, (h3, _) = self.lstm3(out)  # h4形状为 (num_layers, 窗口数(batch_size * max_num_window), hidden_size)
        h3 = h3[-1].view(batch_size, max_num_window, -1)  # 最后一个时间步的隐藏状态
        out = self.dropout4(h3)
        out = out.reshape(-1, 70)  # 将形状调整为 (batch_size * max_num_window, hidden_size)
        out = self.batchnorm3(out)
        out = out.reshape(batch_size, max_num_window, 70)  # 将形状调整回 (batch_size, max_num_window, hidden_size)

        # 通过第四层LSTM
        _, (h4, _) = self.lstm4(out)  # h4形状为 (num_layers, batch_size, hidden_size)
        h4 = h4[-1]
        out = self.dropout3(h4)
        out = out.reshape(-1, 80)  # 将形状调整为 (batch_size, hidden_size)
        out = self.batchnorm4(out)

        # 通过全连接层
        output = self.fc(out)

        # 添加Softmax激活函数
        output = torch.softmax(output, dim=1)
        return output

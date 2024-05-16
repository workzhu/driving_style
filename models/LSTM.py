import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        # LSTM层参数
        self.lstm1 = nn.LSTM(input_size=configs.enc_in,
                             hidden_size=configs.d_model,
                             num_layers=configs.e_layers,
                             batch_first=True,
                             dropout=configs.dropout)

        self.lstm2 = nn.LSTM(input_size=configs.d_model,
                             hidden_size=configs.d_model,
                             num_layers=configs.e_layers,
                             batch_first=True,
                             dropout=configs.dropout)
        # 全连接层用于分类
        self.fc = nn.Linear(configs.d_model, 1)

    def forward(self, packed_input):
        # 解包PackedSequence
        padded_input, input_lengths = pad_packed_sequence(packed_input, batch_first=True)

        batch_size, max_num_window, window_size, _ = padded_input.size()

        # 重塑x以适应第一层LSTM的输入形状
        x = padded_input.reshape(batch_size * max_num_window, window_size, -1)

        # 通过第一层LSTM
        _, (h1, _) = self.lstm1(x)

        # 重塑h1以适应第二层LSTM的输入形状
        h1 = h1[-1].view(batch_size, max_num_window, -1)

        # 通过第二层LSTM
        _, (h2, _) = self.lstm2(h1)
        # 使用最后一个时间步的隐藏状态
        h2 = h2[-1]
        # 通过全连接层
        output = self.fc(h2)

        # 添加Sigmoid激活函数
        output = torch.sigmoid(output)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # LSTM 部分
        self.lstm = nn.LSTM(configs.enc_in, 128, 1, batch_first=True)

        # FCN 部分
        self.conv1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5,
                               padding='same')
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,
                               padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接分类层
        self.fc = nn.Linear(128 + 128, 3)

    def forward(self, x):

        # LSTM 部分
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # FCN 部分
        x = x.permute(0, 2, 1)  # 将输入调整为 [batch_size, input_dim, seq_len]
        conv_out = F.relu(self.bn1(self.conv1(x)))
        conv_out = F.relu(self.bn2(self.conv2(conv_out)))
        conv_out = F.relu(self.bn3(self.conv3(conv_out)))
        conv_out = self.global_avg_pool(conv_out).squeeze(-1)  # 全局平均池化

        # 组合 LSTM 和 FCN 输出
        combined_out = torch.cat((lstm_out, conv_out), dim=1)

        # 全连接层分类
        output = self.fc(combined_out)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_same_padding(kernel_size, stride=1):
    # Formula to calculate the padding that keeps the output size same as input size
    return (kernel_size - stride) // 2

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, 1))

    def forward(self, x):
        # x shape: (batch_size, seq_length, feature_dim)
        scores = torch.matmul(x, self.attention_weights).squeeze(-1)  # (batch_size, seq_length)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_length, 1)
        weighted = torch.mul(x, attn_weights)  # Apply attention weights
        output = torch.sum(weighted, dim=1)  # Sum over the sequence
        return output


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()


        self.conv1 = nn.Conv1d(configs.enc_in, 20, kernel_size=10, stride=1, padding=(10 - 1) // 2)
        self.conv2 = nn.Conv1d(20, 40, kernel_size=5, stride=1, padding=(5 - 1) // 2)
        self.conv3 = nn.Conv1d(40, 80, kernel_size=3, stride=1, padding=(3 - 1) // 2)
        self.lstm1 = nn.LSTM(80, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.attention = AttentionLayer(64)  # 使用自定义注意力层
        self.output = nn.Linear(64, 3)  # 输出层

    def forward(self, x):
        # 调整x的形状为(batch_size, channels, time_steps)
        x = x.transpose(1, 2)  # 这行代码将维度从(batch, time_steps, channels)调整为(batch, channels, time_steps)
        x = F.elu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.15)
        x, (hn, cn) = self.lstm1(x.transpose(1, 2))
        x, (hn, cn) = self.lstm2(x)
        x = self.attention(x)
        x = self.output(x)
        return x


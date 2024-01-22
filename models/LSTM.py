import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        # LSTM层参数
        self.lstm = nn.LSTM(input_size=configs.enc_in,
                            hidden_size=configs.d_model,
                            num_layers=configs.e_layers,
                            batch_first=True,
                            dropout=configs.dropout)

        # 全连接层用于分类
        self.projection = nn.Linear(configs.d_model, configs.num_class)

    def forward(self, x_enc, window_features_enc, padding_mask, x_dec=None, x_mark_dec=None):
        # LSTM层处理时间序列数据
        lstm_out, (h_n, c_n) = self.lstm(x_enc)

        # 只取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 通过全连接层进行分类
        output = self.projection(lstm_out)

        return output

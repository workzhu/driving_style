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

        # 窗口特征处理层
        self.window_feature_processor = nn.Linear(configs.window_feature_dim, 128)

        # 全连接层用于分类
        self.projection = nn.Linear(configs.d_model + 128, configs.num_class)

    def forward(self, x_enc, window_features_enc, padding_mask, x_dec=None, x_mark_dec=None):
        # LSTM层处理时间序列数据
        lstm_out, (h_n, c_n) = self.lstm(x_enc)

        # 只取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 处理窗口特征
        processed_window_features = self.window_feature_processor(window_features_enc)

        # 结合LSTM输出和窗口特征
        combined_features = torch.cat([lstm_out, processed_window_features], dim=1)

        # 通过全连接层进行分类
        output = self.projection(combined_features)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class SampleAttention(nn.Module):
    def __init__(self, input_dim, num_windows):
        super(SampleAttention, self).__init__()
        self.attention_weight = nn.Linear(input_dim, 1)
        self.num_windows = num_windows

    def forward(self, x):
        # x: (batch_size, num_windows, input_dim)
        attention_scores = self.attention_weight(x)  # (batch_size, num_windows, 1)
        attention_scores = F.softmax(attention_scores, dim=1)

        weighted_features = torch.sum(x * attention_scores, dim=1)  # (batch_size, input_dim)
        return weighted_features


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()

        # 时序长度
        self.seq_len = configs.seq_len

        # 预测长度 0
        self.pred_len = configs.pred_len

        # top_k ？超参数，傅里叶变换取前k个频率分量的强度最大的频率
        self.k = configs.top_k

        # parameter-efficient design 卷积层
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    # 定义前向传播函数
    def forward(self, x):
        # 获取输入数据的尺寸：批量大小、时间步长和特征维度
        B, T, N = x.size()

        # 使用傅里叶变换提取前k个重要的频率分量
        period_list, period_weight = FFT_for_Period(x, self.k)

        # 初始化结果列表
        res = []

        # 遍历每个频率分量
        for i in range(self.k):

            # 获取当前频率分量的周期
            period = period_list[i]

            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape
            # 重新整形并应用卷积
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        # 最长时序长度(即每个窗口的长度)
        self.seq_len = configs.seq_len

        # 用于预测的标签长度
        self.label_len = configs.label_len

        # 预测长度 = 0
        self.pred_len = configs.pred_len

        self.window_feature_dim = configs.window_feature_dim

        # 模型由多个TimesBlock组成
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # 这是一个嵌入层，用于将原始数据转换为更适合神经网络处理的形式。
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        self.window_feature_processor = nn.Linear(self.window_feature_dim, 128)

        self.projection = nn.Linear(
            configs.d_model * configs.seq_len + 128, configs.num_class)

    def classification(self, x_enc, window_features_enc, x_mark_enc):

        # embedding
        # 执行enc_embedding的forward函数
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        # TimesNet
        for i in range(self.layer):
            # 执行TimesBlock的forward函数
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        # Output
        # 处理最终的输出
        # 首先应用激活函数
        output = self.act(enc_out)
        # 然后应用dropout层
        output = self.dropout(output)
        # zero-out padding embeddings
        # 零化填充部分的嵌入，以忽略无效数据
        output = output * x_mark_enc.unsqueeze(-1)

        # 处理窗口特征
        processed_window_features = self.window_feature_processor(window_features_enc)

        # 重新整形输出，准备进行分类
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)

        # 将时间序列特征和处理后的窗口特征结合
        combined_features = torch.cat([output, processed_window_features], dim=1)

        # 通过全连接层进行分类
        output = self.projection(combined_features)  # (batch_size, num_classes)
        return output

    def find_nan_indices(self, tensor):
        # 返回一个布尔张量，其中 NaN 为 True
        nan_mask = torch.isnan(tensor)

        # 使用 torch.where 来找到 NaN 的索引
        nan_indices = torch.where(nan_mask)

        # 返回 NaN 的索引
        return nan_indices


    def forward(self, x_enc, window_featrues_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 假设x_enc, window_featrues_enc, x_mark_enc是PyTorch张量

        # 检查输入是否有nan
        if torch.isnan(x_enc).any():
            print("NaN detected in x_enc")
        if torch.isnan(window_featrues_enc).any():
            print("NaN detected in window_features_enc")

            # 假设 window_features_enc 是你的模型中的一个张量
            nan_indices = self.find_nan_indices(window_featrues_enc)
            print("NaN indices in window_features_enc:", nan_indices)
        if torch.isnan(x_mark_enc).any():
            print("NaN detected in x_mark_enc")

        # classification(batch_x, padding_mask)
        dec_out = self.classification(x_enc, window_featrues_enc, x_mark_enc)
        return dec_out  # [B, N]


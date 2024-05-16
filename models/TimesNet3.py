import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from torch.nn import MultiheadAttention


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


class TimesBlock(nn.Module):

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = 100
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()

        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)

            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x

        return res


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # x: (batch_size, seq_length, embed_size)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, V)
        return out


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = 100
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.dropout1 = nn.Dropout(configs.dropout)

        self.BN1 = nn.BatchNorm1d(configs.d_model)

        self.FC = nn.Linear(configs.d_model * self.seq_len, 3)

        self.BN2 = nn.BatchNorm1d(3)

        self.dropout2 = nn.Dropout(configs.dropout)

        self.lstm2 = nn.LSTM(input_size=3,
                             hidden_size=3,
                             num_layers=configs.e_layers,
                             dropout=0.1,
                             batch_first=True)

        self.dropout3 = nn.Dropout(configs.dropout)

        self.projection = nn.Linear(
            3, configs.num_class)

        # 使用自注意力机制
        self.self_attn = SelfAttention(embed_size=configs.d_model)

    def classification(self, bitch_size, all_windows, window_nums):

        # embedding
        enc_out = self.enc_embedding(all_windows, None)  # [B,T,C]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 自注意力
        # attn_out = self.self_attn(enc_out)

        output = self.BN1(enc_out.transpose(1, 2)).transpose(1, 2)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = F.gelu(output)

        output = self.dropout1(output)

        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)

        output = self.FC(output)

        output = self.BN2(output)

        output = F.gelu(output)

        output = self.dropout2(output)

        # 提取每个样本的隐藏状态
        sample_hidden_states = []

        for i in range(bitch_size):
            start_index = i * window_nums
            end_index = (i + 1) * window_nums - 1
            sample_hidden_state = output[start_index:end_index, :]
            sample_hidden_states.append(sample_hidden_state)

        output = torch.stack(sample_hidden_states)

        # 第二层LSTM
        lstm_out, (_, _) = self.lstm2(output)

        # 取最后一个时间步的输出作为最终输出
        lstm_last_output = lstm_out[:, -1, :]

        # 分类层
        output = self.projection(lstm_last_output)  # Stack and classify

        return output  # [B, N]

    def forward(self, x):

        B, T, N = x.size()

        stride = 50  # 滑动窗口的步长
        window_size = self.seq_len

        window_nums = (T - window_size + stride) // stride

        all_windows = []
        for i in range(0, T - window_size + 1, stride):
            window = x[:, i:i + window_size, :]
            all_windows.append(window)

        all_windows = torch.stack(all_windows, dim=0)
        all_windows = all_windows.permute(1, 0, 2, 3).contiguous()
        all_windows = all_windows.view(-1, window_size, N)

        dec_out = self.classification(B, all_windows, window_nums)
        return dec_out  # [B, N]



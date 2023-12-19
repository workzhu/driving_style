import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


# 定义一个继承自nn.Module的PositionalEmbedding类，用于生成位置嵌入
# 这个PositionalEmbedding类生成了一个固定大小的位置编码矩阵，并在模型的前向传播过程中将其与输入数据相结合。
# 这种方法使得模型可以识别序列数据中每个时间点的位置，有助于处理缺乏自然顺序感知能力的模型结构，如Transformer。
# 此实现中，位置编码使用了正弦和余弦函数的交替组合来生成，允许模型捕获不同时间点之间的相对位置关系。
class PositionalEmbedding(nn.Module):

    # 类的初始化函数，接收嵌入向量的维度d_model，最大长度max_len
    def __init__(self, d_model, max_len=80000):
        # 调用父类的初始化函数
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        # 计算位置编码，只在log空间中计算一次
        # 初始化一个大小为(max_len, d_model)的位置编码矩阵，全部填充为0，并确保数据类型为浮点型
        pe = torch.zeros(max_len, d_model).float()
        # 设置位置编码不需要梯度计算
        pe.require_grad = False

        # 生成一个从0到max_len-1的位置索引，并增加一个维度，变为(max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 生成一个除数项，用于计算位置编码的分母，这里使用的是指数函数来产生一个平滑的减小
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 使用正弦函数计算位置编码的偶数部分
        pe[:, 0::2] = torch.sin(position * div_term)

        # 使用余弦函数计算位置编码的奇数部分
        pe[:, 1::2] = torch.cos(position * div_term)

        # 调整位置编码矩阵的维度，使其适合模型的输入
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为模型的缓冲区，从而使它成为模型的一部分，这样在保存模型时，位置编码矩阵也会被保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 返回位置编码，根据输入x的长度截取相应的位置编码
        return self.pe[:, :x.size(1)]


# 定义一个继承自nn.Module的TokenEmbedding类，用于将原始特征转换为嵌入向量
# 目的：将原始时间序列数据转换成嵌入向量，这有助于模型更好地理解和处理这些数据
# 实现：使用了卷积神经网络（nn.Conv1d），这种选择表明它旨在捕捉时间序列数据中的局部特征。
# 这个TokenEmbedding类使用一维卷积网络来处理时间序列数据。
# 它的目的是从原始数据中提取有用的特征，并将这些特征转换为嵌入向量，从而为后续的模型处理做好准备。
# 卷积操作可以捕捉到时间序列中的局部依赖关系，而He初始化方法有助于保持激活函数的输出分布稳定，从而有利于模型的学习和收敛。
class TokenEmbedding(nn.Module):

    # c_in：输入通道数，即原始特征的维度。
    # d_model：输出通道数，嵌入后的特征维度。
    def __init__(self, c_in, d_model):

        # 调用父类的构造函数
        super(TokenEmbedding, self).__init__()

        # 根据PyTorch的版本选择适当的padding值
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # 定义一个一维卷积层，用于提取时间序列特征。输入通道数为c_in，输出通道数为d_model，
        # 核大小为3，使用循环填充模式，无偏置项
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        # 遍历模型中的所有模块
        for m in self.modules():
            # 如果模块是一维卷积层
            if isinstance(m, nn.Conv1d):
                # 使用He初始化（也称为Kaiming初始化），适用于ReLU激活函数的变体，如leaky ReLU
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    # 定义前向传播函数
    def forward(self, x):

        # 将输入x的维度进行调整，以匹配卷积层的输入要求（将特征维度放到通道维度上），
        # 应用卷积操作后再将维度调回原来的顺序
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


# 定义一个继承自nn.Module的TemporalEmbedding类，用于生成时间相关的嵌入
# 类的初始化函数，接收嵌入向量的维度d_model，嵌入类型embed_type和时间频率freq

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        # 调用父类的初始化函数
        super(TemporalEmbedding, self).__init__()

        # 定义不同时间单位的大小
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # 根据嵌入类型选择嵌入类
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


# 定义一个继承自nn.Module的TimeFeatureEmbedding类，用于生成基于时间特征的嵌入
# TimeFeatureEmbedding类使用一个线性层来将基于时间的特征转换为嵌入向量。
# 它根据时间频率（如小时、分钟等）的不同，选择适当的输入维度。
# 这个类使模型能够利用来自原始时间序列数据的时间特征，如时间点的小时数或星期几等，从而帮助模型更好地理解和处理时间序列数据。
# 通过这种方式，模型可以学习到时间特征与目标变量之间的关系，这对于时间序列预测或分类任务来说是非常有用的。
class TimeFeatureEmbedding(nn.Module):

    # 类的初始化函数，接收嵌入向量的维度d_model，嵌入类型embed_type，以及时间频率freq
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        # 调用父类的初始化函数
        super(TimeFeatureEmbedding, self).__init__()

        # 定义一个频率映射字典，用于确定不同时间频率对应的输入维度
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

        # 根据提供的时间频率freq从映射中获取输入维度d_inp
        d_inp = freq_map[freq]

        # 定义一个线性层，用于将时间特征转换为嵌入向量，输入维度为d_inp，输出维度为d_model，无偏置项
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 值嵌入(特征维度, 模型维度
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # 位置嵌入
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 时间嵌入  这里执行TimeFeatureEmbedding
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        # 定义一个dropout层，用于正则化
        self.dropout = nn.Dropout(p=dropout)

    # 定义前向传播函数，接收输入数据x和可选的时间标记x_mark
    def forward(self, x, x_mark):
        # 如果没有提供时间标记，则只使用值嵌入和位置嵌入
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        # 否则，使用值嵌入、时间嵌入和位置嵌入
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # 应用dropout层并返回结果
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

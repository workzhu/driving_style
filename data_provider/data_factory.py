from torch.utils.data.dataset import T_co

from data_provider.uea import collate_fn, padding_mask
from torch.utils.data import DataLoader
from data_provider.MyDataLoader import MyDataLoader
from data_provider.uea import subsample, interpolate_missing, Normalizer
import copy
from torch.utils.data import Dataset
import torch
import numpy as np


class Window:
    def __init__(self, window_df, label=None, sample_id=None, padding_masks=None):
        self.window_df = window_df
        self.label = label
        self.sample_id = sample_id
        self.padding_masks = padding_masks
        self.window_features = self.extract_features_from_window()

    def extract_features_from_window(self):
        """
        从每个窗口提取特征。

        参数:
        windows (list of DataFrame): 包含多个窗口的列表，每个窗口是一个DataFrame。

        返回:
        DataFrame: 包含每个窗口特征统计的DataFrame。
        """
        features = {}

        features_name = ['x_acc_clean', 'y_acc_clean', 'z_acc_clean',
                         'x_mag', 'y_mag', 'z_mag',
                         'x_gra', 'y_gra', 'z_gra',
                         'x_gyro', 'y_gyro', 'z_gyro',
                         'speed', 'bearing']


        # 遍历每一列并提取特征
        for feature in features_name:
            max_value = self.window_df[feature].max()
            min_value = self.window_df[feature].min()
            mean_value = self.window_df[feature].mean()
            var_value = self.window_df[feature].var()
            skew_value = self.window_df[feature].skew()
            if len(self.window_df[feature].dropna()) >= 4 and self.window_df[feature].var() != 0:
                kurt_value = self.window_df[feature].kurt()
            else:
                kurt_value = 0  # 或其他合适的默认值
            std_dev_value = self.window_df[feature].std()

            features[f'max_{feature}'] = max_value
            features[f'min_{feature}'] = min_value
            features[f'mean_{feature}'] = mean_value
            features[f'var_{feature}'] = var_value
            features[f'skew_{feature}'] = skew_value
            features[f'kurt_{feature}'] = kurt_value
            features[f'std_dev_{feature}'] = std_dev_value
        return features


class WindowedDataset(Dataset):
    def __getitem__(self, ind):
        window = self.windows[ind]
        return window

    def __len__(self):
        """
        返回数据集中窗口的数量。
        """

        return len(self.windows)

    def __init__(self, dataset, window_size, step_size):
        self.dataset = dataset
        self.windows = self.create_sliding_windows(window_size, step_size)
        self.window_size = window_size
        self.step_size = step_size

        print('窗口长度：', self.window_size)
        print('窗口步长：', self.step_size)
        print('窗口数量：', len(self.windows))

    def is_maneuver(self, window_df, length):
        """
            计算 DataFrame 中陀螺仪数据的 F1 值，并判断是否小于阈值。

            参数:
            df (pd.DataFrame): 包含陀螺仪数据的 DataFrame，应有 'g_x', 'g_y', 'g_z' 列。
            threshold (float): 与 F1 值进行比较的阈值。

            返回:
            bool: 如果 F1 值小于等于阈值，则为 True，否则为 False。
        """
        F1 = []
        for i in range(length):
            F1.append(np.sqrt(window_df['x_gyro'].iloc[i] ** 2
                              + window_df['y_gyro'].iloc[i] ** 2
                              + window_df['z_gyro'].iloc[i] ** 2))
        # 计算 F1 平均值
        F1_avg = np.mean(F1)

        return F1_avg >= 0.1

    def create_sliding_windows(self, window_size, step_size):
        """
        在 DataFrame 的每个样本内创建滑动窗口。

        参数:
        window_size (int): 每个窗口的大小。
        step_size (int): 创建下一个窗口时向前移动的步数。

        返回:
        windows (list): 包含所有窗口的列表。
        samples (list): 包含每个窗口对应的样本索引的列表。
        padding_masks (list): 每个窗口的填充掩码列表。
        """
        windows = []
        for sample_index in self.dataset.feature_df.index.unique():
            sample_df = self.dataset.feature_df.loc[sample_index]

            # 处理样本长度小于窗口长度的情况
            if len(sample_df) < window_size:
                length = len(sample_df)

                if self.is_maneuver(sample_df, length) and (length * 0.02 >= 1):
                    window = Window(sample_df)
                    window.label = torch.from_numpy(self.dataset.labels_df.loc[sample_index].values)
                    window.sample_id = sample_index
                    windows.append(window)
                    window.real_length = length

                continue

            # 创建滑动窗口
            for start in range(0, len(sample_df), step_size):
                end = start + window_size
                if end <= len(sample_df):
                    window_df = sample_df.iloc[start:end]
                    length = window_size
                else:
                    window_df = sample_df.iloc[start:]
                    # 处理样本长度不是步长整数倍的情况
                    length = len(sample_df.iloc[start:])

                if self.is_maneuver(window_df, length):
                    window = Window(window_df)
                    window.label = torch.from_numpy(self.dataset.labels_df.loc[sample_index].values)
                    window.sample_id = sample_index
                    window.real_length = length
                    windows.append(window)
        return windows


def make_dataset(data_set, df, name_str, window_size, step_size, limit_size=None):
    temp_set = copy.deepcopy(data_set)
    temp_set.all_df = df
    temp_set.all_IDs = df.index.unique()
    print(name_str, list(temp_set.all_IDs))

    if limit_size is not None:

        # 如果是整数，那么就是样本数
        if limit_size > 1:
            limit_size = int(limit_size)

        # 如果是小数，那么就是比例
        else:  # interpret as proportion if in (0, 1]

            # 样本数 = 比例 * 样本总数
            limit_size = int(limit_size * len(temp_set.all_IDs))
        # 按限制的样本数截取样本ID
        temp_set.all_IDs = temp_set.all_IDs[:limit_size]
        # 根据截取的样本ID截取数据集
        temp_set.all_df = temp_set.all_df.loc[temp_set.all_IDs]

    # use all features
    temp_set.feature_names = temp_set.all_df.columns

    # 选择特征
    temp_set.feature_df = temp_set.all_df

    # 预处理
    # 标准化s
    normalizer = Normalizer()

    temp_set.feature_df = normalizer.normalize(temp_set.feature_df)

    temp_windowed_set = WindowedDataset(temp_set, window_size, step_size)

    return temp_windowed_set


def data_provider(args):
    batch_size = args.batch_size  # bsz for train and valid
    window_size = args.window_size
    step_size = args.step_size
    drop_last = False

    # 读数据（对此类进行改写）
    data_set = MyDataLoader(
        root_path=args.root_path

    )

    train_df, test_df, vali_df = data_set.split_by_ratio(ratio=0.8)

    train_set = make_dataset(data_set, train_df, '训练集行程ID：', window_size, step_size)
    test_set = make_dataset(data_set, test_df, '测试集行程ID：', window_size, step_size)
    vali_set = make_dataset(data_set, vali_df, '验证集行程ID：', window_size, step_size)

    # 结合了数据集和采样器，提供了对数据集的读取操作，
    # 还提供了单进程或多进程迭代器，
    # 还提供了数据打乱和重排序的功能，shuffle
    # 还提供了对数据批量处理的功能，batch_size
    # 还提供了对数据预取的功能。
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,  # 每个batch的大小
        shuffle=True,  # 是否打乱数据
        num_workers=args.num_workers,  # 读取数据的线程数
        drop_last=drop_last,  # 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch

        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,  # 每个batch的大小
        shuffle=False,  # 是否打乱数据
        num_workers=args.num_workers,  # 读取数据的线程数
        drop_last=drop_last,  # 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )

    vali_loader = DataLoader(
        vali_set,
        batch_size=batch_size,  # 每个batch的大小
        shuffle=False,  # 是否打乱数据
        num_workers=args.num_workers,  # 读取数据的线程数
        drop_last=drop_last,  # 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )

    return train_set, train_loader, test_set, test_loader, vali_set, vali_loader


# Unit test
"""
import argparse
args = argparse.Namespace()
args.num_workers = 0
args.freq = 'h'
args.batch_size = 64
args.task_name = 'classification'
args.root_path = 'D:\实验数据'

data_provider(args, 'train')
"""

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):

        # 数据集的根目录
        self.root_path = root_path

        # 读取数据集
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)

    # 读取数据集
    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        # 选择训练和评估的路径

        # 如果没有指定文件列表，那么就是所有的文件
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths

        # 如果指定了文件列表，那么就是指定的文件
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]

        # 如果没有文件，那么就报错
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))

        # 如果指定了flag，那么就只保留指定的文件
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))

        # 这里是.ts类型的后缀文件
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern = '*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        # 读取数据集
        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    # 读取单个数据集
    def load_single(self, filepath):

        # 读取.ts文件——>dataframe
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        # df的每格是一个时间序列

        # 读取标签(这里应为可能是多标签，所以用Series)
        labels = pd.Series(labels, dtype="category")

        # 获取所有的类别
        self.class_names = labels.cat.categories

        # 将标签转换的编码转化为df
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        # df每个单元格的size,即每个时间序列的长度
        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        #  每列与第一列长度的差值的绝对值
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # 如果有差值，那么就进行插值
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions

            # 工作方式是对df中的每个元素调用subsample函数，其中该元素的值被作为y传递给subsample。
            df = df.applymap(subsample)

        # 如果没有差值，那么就不用插值
        lengths = df.applymap(lambda x: len(x)).values

        # 每列与第一格长度是否相等（即是否为等长序列）
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))

        # 获取最大序列长度
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
            torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)

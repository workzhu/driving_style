from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Exp_Tsne(Exp_Basic):

    # 初始化模型
    def _build_model(self):

        # 获取序列最长长度
        self.args.seq_len = self.args.window_size

        self.args.pred_len = 0

        # 获取特征维度
        self.args.enc_in = self.train_data.dataset.feature_df.shape[1]

        # 获取窗口特征维度
        self.args.window_feature_dim = len(self.train_data.windows[0].window_features)

        print("window_feature_dim", self.args.window_feature_dim)

        # 获取标签数
        self.args.num_class = len(self.train_data.dataset.class_names)

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def apply_tsne(self, n_components=2, perplexity=30, learning_rate=200):
        """
        应用t-SNE降维并进行可视化。自动使用类中存储的数据。
        :param n_components: t-SNE降维后的维数。
        :param perplexity: t-SNE的困惑度参数。
        :param learning_rate: t-SNE的学习率。
        """
        # 假设 self.train_data 是一个包含特征和标签的数据集
        features = self.train_data.dataset.feature_df  # 需要根据您的数据集调整
        labels = self.train_data.dataset.labels_df    # 需要根据您的数据集调整

        print(features)
        print(labels)

        # 1. 确保索引一致
        feature_df = features.reset_index()
        labels = labels.reset_index()

        print(labels.head())

        feature_df.rename(columns={'index': 'ID'}, inplace=True)
        labels.rename(columns={'index': 'ID'}, inplace=True)
        labels.rename(columns={0: 'label'}, inplace=True)

        # 2. 匹配样本和标签
        # 假设 features 的索引是 ID，并且 labels 也有一个相应的 ID 列
        matched_df = features.merge(labels, left_index=True, right_on='ID')

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        tsne_results = tsne.fit_transform(matched_df.drop('label', axis=1))

        # 可视化
        plt.figure(figsize=(8, 5))
        for label in np.unique(matched_df['label']):
            indices = matched_df['label'] == label
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=str(label), s=1)
        plt.legend()

        # 保存图表到文件
        plt.savefig('tsne_visualization.png')

        # 显示图表（可选）
        plt.show()

    def _get_data(self):
        train_data, train_loader, test_data, test_loader, vali_data, vali_loader = data_provider(self.args)

        return train_data, train_loader, test_data, test_loader, vali_data, vali_loader

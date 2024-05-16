from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import csv

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):

    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    # 初始化模型
    def _build_model(self):

        # 获取序列最长长度
        self.args.seq_len = self.args.window_size

        self.args.pred_len = 0

        # 获取特征维度
        self.args.enc_in = self.dataset.enc_in

        # 获取标签数
        self.args.num_class = self.dataset.num_class

        self.total_samples = sum(self.dataset.label_counts.values())

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self):
        dataset, train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = \
            data_provider(self.args)
        return dataset, train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=0.001)
        return model_optim

    def _select_criterion(self):
        weights = {class_label: self.total_samples / count for class_label, count in self.dataset.label_counts.items()}
        weights_tensor = torch.tensor([weights['normal'], weights['aggressive']])
        print(weights_tensor)
        weights_tensor = weights_tensor.to(self.device)

        # 创建带有惩罚权重的损失函数
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        # criterion = nn.CrossEntropyLoss()
        # criterion = torch.nn.BCELoss()
        # criterion = criterion.to(self.device)
        return criterion

    def validate(self, vali_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            for i, (batch_x, window_features, batch_y) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)

                window_features = window_features.float().to(self.device)

                # 将标签转换为类别索引
                #  true_labels = torch.argmax(batch_y, dim=1)
                true_labels = batch_y.long()
                true_labels = true_labels.to(self.device)

                outputs = self.model(batch_x, window_features)

                loss = criterion(outputs, true_labels)
                # outputs = outputs.squeeze()
                # loss = criterion(outputs, true_labels.float())

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                # predictions = (outputs > 0.5).long()

                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())

        avg_loss = total_loss / len(vali_loader)
        avg_accuracy = accuracy_score(all_true_labels, all_predictions)
        avg_precision = precision_score(all_true_labels, all_predictions, average='macro')
        avg_recall = recall_score(all_true_labels, all_predictions, average='macro')
        avg_f1 = f1_score(all_true_labels, all_predictions, average='macro')

        return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1


    # 训练
    def train(self, setting):

        # 根据设置的参数创建用于存储模型检查点的路径
        path = os.path.join(self.args.checkpoints, setting)

        if not os.path.exists(path):
            # 如果路径不存在，则创建该路径
            os.makedirs(path)

        # 记录当前时间，用于计算训练耗时
        time_now = time.time()

        saved_metrics_list = []

        # Rebuild the model for each fold to reset weights
        self.model = self._build_model().to(self.device)

        # Reset the optimizer
        model_optim = self._select_optimizer()

        # 选择损失函数
        criterion = self._select_criterion()

        # 初始化早停机制，根据设置的耐心值来决定何时停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 打开一个文件用于追加写入指标
        with open("metrics.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(
                ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1'])

        for epoch in range(self.args.train_epochs):

            # 将模型设置为训练模式
            self.model.train()

            # 记录当前周期的开始时间
            epoch_time = time.time()

            total_loss = 0
            total_acc = 0

            # 遍历训练数据加载器中的每个批次
            for i, (batch_x, window_features, batch_y) in enumerate(self.train_loader):

                # 清除之前的梯度
                model_optim.zero_grad()
                # 将数据和标签转移到设备（例如GPU）
                batch_x = batch_x.float().to(self.device)
                window_features = window_features.float().to(self.device)

                # 将标签转换为类别索引
                # true_labels = torch.argmax(batch_y, dim=1)

                true_labels = batch_y.long()
                true_labels = true_labels.to(self.device)

                outputs = self.model(batch_x, window_features)

                loss = criterion(outputs, true_labels)
                # outputs = outputs.squeeze()
                # loss = criterion(outputs, true_labels.float())

                loss.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                model_optim.step()

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                # predictions = (outputs > 0.5).long()

                acc = (predictions == true_labels).float().mean()

                total_acc += acc.item()

            avg_train_loss = total_loss / len(self.train_loader)
            avg_train_acc = total_acc / len(self.train_loader)

            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(self.valid_loader, criterion)

            print(f"Epoch {epoch + 1}/{self.args.train_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

            # 在每轮训练后立即将指标追加到文件
            with open("metrics.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [epoch + 1, avg_train_loss, avg_train_acc, val_loss, val_acc, val_precision, val_recall,
                     val_f1])

            early_stopping(val_f1, self.model, path)

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Training complete in {time.time() - time_now:.2f} seconds")

        return self.model

    # 在一个单独的测试函数中调用
    def test(self, setting, test=0):

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        criterion = self._select_criterion()
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            for i, (batch_x, window_features, batch_y) in enumerate(self.test_loader):

                batch_x = batch_x.float().to(self.device)
                window_features = window_features.float().to(self.device)

                # 将标签转换为类别索引
                # true_labels = torch.argmax(batch_y, dim=1)

                true_labels = batch_y.long()
                true_labels = true_labels.to(self.device)

                outputs = self.model(batch_x, window_features)

                loss = criterion(outputs, true_labels)

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())

        avg_accuracy = accuracy_score(all_true_labels, all_predictions)
        avg_precision = precision_score(all_true_labels, all_predictions, average='macro')
        avg_recall = recall_score(all_true_labels, all_predictions, average='macro')
        avg_f1 = f1_score(all_true_labels, all_predictions, average='macro')

        print(f"Val Acc: {avg_accuracy:.4f}, "
              f"Val Precision: {avg_precision:.4f}, Val Recall: {avg_recall:.4f}, Val F1: {avg_f1:.4f}")

        conf_mat = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=False, cmap="Blues", xticklabels=np.arange(3), yticklabels=np.arange(3),
                    cbar=False, fmt="d")  # 显示数量

        # 计算每个类别的准确率
        class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)

        # 在每个单元格中显示数量和准确率
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(j + 0.5, i + 0.5, f"{conf_mat[i, j]} ({class_accuracy[i] * 100:.2f}%)",
                         ha="center", va="center", color="white", fontsize=10, weight="bold")

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (with Counts and Accuracy)')
        plt.savefig('confusion_matrix.png')  # 保存到安全的路径
        plt.show()

    def randomforest(self):

        train_x = []
        train_y = []

        for i, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            train_x.append(batch_x.cpu().numpy())
            train_y.append(batch_y.cpu().numpy())
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        train_y = np.argmax(train_y, axis=1)

        test_x = []
        test_y = []

        for i, (batch_x, batch_y) in enumerate(self.valid_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            test_x.append(batch_x.cpu().numpy())
            test_y.append(batch_y.cpu().numpy())
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        test_y = np.argmax(test_y, axis=1)

        # 将输入数据重新塑形为2D
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)

        print("train_x shape:", train_x.shape)
        print("train_y shape:", test_x.shape)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        accuracy = accuracy_score(test_y, y_pred)
        print("Accuracy:", accuracy)

        print("Confusion Matrix:")
        print(confusion_matrix(test_y, y_pred))

        print("Normalized Confusion Matrix:")
        print(confusion_matrix(test_y, y_pred, normalize='true'))

        # 绘制混淆矩阵（归一化）
        conf_mat = confusion_matrix(test_y, y_pred, normalize='true')
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, cmap="Blues", xticklabels=np.arange(3), yticklabels=np.arange(3),
                    annot_kws={"size": 10}, fmt=".2f")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.savefig('confusion_matrix.png')  # 保存到安全的路径
        plt.show()

        return clf

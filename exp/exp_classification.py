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

    def _get_data(self):
        train_data, train_loader, test_data, test_loader, vali_data, vali_loader = data_provider(self.args)

        return train_data, train_loader, test_data, test_loader, vali_data, vali_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        # 假设你有一个二分类问题，其中类别 0 的样本数量是类别 1 的 10 倍
        # 设置更高的权重给较少的类别（类别 1）
        # 权重移到相同设备
        weights = torch.tensor([1.0, 0.3])

        # 创建带有惩罚权重的损失函数
        criterion = nn.CrossEntropyLoss(weight=weights)

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        sample_true_labels = {}  # 用于存储每个样本的真实标签

        sample_preds = defaultdict(list)  # 用于存储每个样本的所有预测

        # 这行代码将模型设置为评估模式，通常会关闭诸如dropout等只在训练时使用的特性
        self.model.eval()

        # 在评估模型时不需要计算梯度，这可以减少内存使用并加速计算
        with torch.no_grad():
            # 迭代验证数据
            for i, (batch_x, label, sample_id, window_features, padding_mask) in enumerate(vali_loader):

                # 数据预处理和模型预测
                # 将数据和标签转移到设备（例如GPU）
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                window_features = window_features.float().to(self.device)

                # 此处调用的是模型的forward()方法
                # outputs = self.model(batch_x, window_features, padding_mask, None, None)
                outputs = self.model(batch_x, padding_mask, None, None)

                # 计算损失并收集预测
                pred = outputs.detach().cpu()

                loss = criterion(pred, label.long().squeeze().cpu())
                # loss = criterion(pred, label.long().cpu())

                prediction = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1).cpu().numpy()

                total_loss.append(loss)

                label = label.squeeze().cpu().numpy()
                sample_id = list(sample_id)
                for sid, p, true_label in zip(sample_id, prediction, label):
                    sample_preds[sid].append(p)
                    sample_true_labels[sid] = true_label  # 存储或更新每个样本的真实标签

        # 投票机制，并按照样本 ID 的排序得到最终预测
        final_predictions = [Counter(sample_preds[sid]).most_common(1)[0][0] for sid in sorted(sample_true_labels)]

        # 准确率计算
        trues = [sample_true_labels[sid] for sid in sorted(sample_true_labels)]  # 按照 sample_id 排序的真实标签
        accuracy = np.sum(np.array(final_predictions) == np.array(trues)) / len(trues)

        # 计算总体损失和准确率
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss, accuracy

        '''
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # 重置模型为训练模式
        self.model.train()
        return total_loss, accuracy
        '''

    # 训练
    def train(self, setting):

        # 根据设置的参数创建用于存储模型检查点的路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            # 如果路径不存在，则创建该路径
            os.makedirs(path)
        # 记录当前时间，用于计算训练耗时
        time_now = time.time()

        # 获取训练数据的批次数量
        train_steps = len(self.train_loader)
        # 初始化早停机制，根据设置的耐心值来决定何时停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 选择优化器
        model_optim = self._select_optimizer()
        # 选择损失函数
        criterion = self._select_criterion()

        epoch_count = 0
        train_losses = []
        val_losses = []
        train_acces = []
        val_acces = []

        # 对于每一个训练周期
        for epoch in range(self.args.train_epochs):
            # 训练周期计数加1
            epoch_count = epoch + 1

            # 初始化迭代计数和训练损失列表
            iter_count = 0
            train_loss = []

            sample_true_labels = {}  # 用于存储每个样本的真实标签

            sample_preds = defaultdict(list)  # 用于存储每个样本的所有预测

            # 将模型设置为训练模式
            self.model.train()
            # 记录当前周期的开始时间
            epoch_time = time.time()

            # 遍历训练数据加载器中的每个批次
            for i, (batch_x, label, sample_id, window_features, padding_mask) in enumerate(self.train_loader):

                # 迭代计数加1
                iter_count += 1
                # 清除之前的梯度
                model_optim.zero_grad()
                # 将数据和标签转移到设备（例如GPU）
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                window_features = window_features.float().to(self.device)

                criterion = criterion.to(self.device)

                # 通过模型获取预测输出
                # outputs = self.model(batch_x, window_features, padding_mask, None, None)
                outputs = self.model(batch_x, padding_mask, None, None)

                # 计算损失并添加到损失列表
                loss = criterion(outputs, label.long().squeeze(-1))

                prediction = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1).cpu().numpy()

                train_loss.append(loss.item())

                label = label.squeeze().cpu().numpy()
                sample_id = list(sample_id)
                for sid, p, true_label in zip(sample_id, prediction, label):
                    sample_preds[sid].append(p)
                    sample_true_labels[sid] = true_label  # 存储或更新每个样本的真实标签

                # 每100次迭代打印一次进度
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播，梯度裁剪，参数更新
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            # 打印当前周期的耗时和平均损失
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 投票机制，并按照样本 ID 的排序得到最终预测
            final_predictions = [Counter(sample_preds[sid]).most_common(1)[0][0] for sid in sorted(sample_true_labels)]

            # 准确率计算
            trues = [sample_true_labels[sid] for sid in sorted(sample_true_labels)]  # 按照 sample_id 排序的真实标签
            train_accuracy = np.sum(np.array(final_predictions) == np.array(trues)) / len(trues)
            train_loss = np.average(train_loss)

            val_loss, val_accuracy = self.vali(self.vali_data, self.vali_loader, criterion.to('cpu'))
            test_loss, test_accuracy = self.vali(self.test_data, self.test_loader, criterion.to('cpu'))

            train_losses.append(train_loss)

            val_losses.append(val_loss)

            train_acces.append(train_accuracy)

            val_acces.append(val_accuracy)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Vali Loss: {4:.3f} Vali Acc: {5:.3f} Test Loss: {6:.3f} Test Acc: {7:.3f}"
                .format(epoch + 1, train_steps, train_loss, train_accuracy, val_loss, val_accuracy, test_loss,
                        test_accuracy))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # if (epoch + 1) % 5 == 0:
        # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = folder_path + 'loss_and_accuracy.csv'

        # 创建DataFrame
        df = pd.DataFrame({
            'Epoch': range(1, epoch_count + 1),
            'Train Loss': train_losses,
            'Validation Loss': val_losses,
            'Train Accuracy': train_acces,
            'Validation Accuracy': val_acces
        })

        # 将DataFrame保存为CSV
        df.to_csv(file_name, index=False)

        return self.model

    def test(self, setting, test=0):

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        sample_true_labels = {}  # 用于存储每个样本的真实标签

        sample_preds = defaultdict(list)  # 用于存储每个样本的所有预测
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, sample_id, window_features, padding_mask) in enumerate(self.test_loader):

                # 数据预处理和模型预测
                # 将数据和标签转移到设备（例如GPU）
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                window_features = window_features.float().to(self.device)

                # outputs = self.model(batch_x, window_features, padding_mask, None, None)
                outputs = self.model(batch_x, padding_mask, None, None)

                prediction = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1).cpu().numpy()

                label = label.squeeze().cpu().numpy()
                sample_id = list(sample_id)
                for sid, p, true_label in zip(sample_id, prediction, label):
                    sample_preds[sid].append(p)
                    sample_true_labels[sid] = true_label  # 存储或更新每个样本的真实标签

        print("sample_true_labels", sample_true_labels)
        print("sample_preds", sample_preds)

        # 投票机制，并按照样本 ID 的排序得到最终预测
        final_predictions = [Counter(sample_preds[sid]).most_common(1)[0][0] for sid in sorted(sample_true_labels)]

        # 准确率计算
        trues = [sample_true_labels[sid] for sid in sorted(sample_true_labels)]  # 按照 sample_id 排序的真实标签
        accuracy = np.sum(np.array(final_predictions) == np.array(trues)) / len(trues)

        # 打印每个样本的信息
        for sid, vote_result in zip(sorted(sample_true_labels), final_predictions):
            true_label = sample_true_labels[sid]
            is_correct = '正确' if vote_result == true_label else '错误'
            true_label = '正常' if true_label == 1 else '激进'
            vote_result = '正常' if vote_result == 1 else '激进'
            print(f'行程ID: {sid}, 行程标签: {true_label}, Voted Prediction: {vote_result}, {is_correct}')

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        # 打印每个样本的信息
        for sid, vote_result in zip(sorted(sample_true_labels), final_predictions):
            true_label = sample_true_labels[sid]
            is_correct = '正确' if vote_result == true_label else '错误'
            true_label = '正常' if true_label == 1 else '激进'
            vote_result = '正常' if vote_result == 1 else '激进'
            f.write(f'行程ID: {sid}, 行程标签: {true_label}, Voted Prediction: {vote_result}, {is_correct}')
            f.write('\n')
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return

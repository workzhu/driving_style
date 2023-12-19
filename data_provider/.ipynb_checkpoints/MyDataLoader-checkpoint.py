import os
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import quaternion
from scipy.spatial.transform import Rotation as R
from pykalman import KalmanFilter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import glob
import re
from torch.utils.data import Dataset
from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings

warnings.filterwarnings('ignore')


class MyDataLoader(Dataset):
    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
            torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):

        self.max_seq_len = 0
        self.class_names = 0
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)

    def load_all(self, root_path, file_list=None, flag=None):

        data_list = []
        label_list = []

        # 遍历所有驾驶员
        for driver in os.listdir(root_path):
            driver_dir = os.path.join(root_path, driver)

            # 遍历每个驾驶员的所有行程
            for trip in os.listdir(driver_dir):
                trip_dir = os.path.join(driver_dir, trip)

                # 检查当前项是否为目录，如果不是，则跳过
                if not os.path.isdir(trip_dir):
                    continue

                print("**********************************************************")
                print("读取文件夹：", trip)

                if os.path.exists(trip_dir + "/clean_data.csv"):
                    print("文件夹：", trip, "已经存在清洗后的数据！")
                    temp_data = pd.read_csv(os.path.join(trip_dir, 'clean_data.csv'))
                else:
                    # 读取每个行程的CSV文件
                    accelerometer = self.read_with_template(os.path.join(trip_dir, 'Accelerometer.csv'),
                                                            ['time', 'seconds_elapsed', 'x', 'y', 'z'])

                    orientation = self.read_with_template(os.path.join(trip_dir, 'Orientation.csv'),
                                                          ['time', 'qw', 'qx', 'qy', 'qz', 'roll', 'pitch', 'yaw'])

                    gravity = self.read_with_template(os.path.join(trip_dir, 'Gravity.csv'), ['time', 'x', 'y', 'z'])

                    magnetometer = self.read_with_template(os.path.join(trip_dir, 'Magnetometer.csv'),
                                                           ['time', 'x', 'y', 'z'])

                    gps = self.read_with_template(os.path.join(trip_dir, 'Location.csv'),
                                                  ['time', 'bearing', 'bearingAccuracy', 'speed',
                                                   'speedAccuracy', 'latitude', 'longitude'])

                    temp_data = self.merge_data(accelerometer, gravity, magnetometer, gps, orientation)

                    # 重定向
                    self.reorientation(temp_data)

                    # 对加速度进行卡尔曼滤波
                    self.data_KalmanFilter3D(temp_data, 1e-6, 1e-5)

                    # 保存清洗后的数据
                    temp_data.to_csv(os.path.join(trip_dir, 'clean_data.csv'), index=False)

                    # self.acc_clean_figure(temp_data)

                # 删除多个与时间相关的列
                time_related_columns = ['time', 'seconds_elapsed']  # 这里添加所有您想删除的时间相关列的名称

                # 假设 dataList 是一个包含多个 DataFrame 的列表
                temp_data = temp_data.drop(time_related_columns, axis=1)

                if temp_data.isnull().any().any():
                    print("Warning: There are missing values in the data")
                    # 打印出含有空值的列
                    print("Columns with missing values:", temp_data.columns[temp_data.isnull().any()])

                    # 使用上一个值填充空值
                    temp_data.fillna(method='ffill', inplace=True)

                data_list.append(temp_data)

                # 提取行程标签
                trip_label = trip.split('_')[2]  # 假设标签是文件夹名称的第三个元素，且normal为0，abnormal为1

                label_list.append(trip_label)
                
                print("行程ID：", len(label_list)-1, "    行程标签：", trip_label)

                print("**********************************************************")

        print("数据读取结束！共读取", len(data_list), "次行程。")

        # 使用enumerate获取每个DataFrame及其在列表中的索引
        # 然后为每个DataFrame设置新的索引，这个索引是其在列表中的位置
        # 最后将这些DataFrame合并成一个大的DataFrame
        data = pd.concat([df.set_index([len(df) * [idx]]) for idx, df in enumerate(data_list)])

        # 确定最大行数
        self.max_seq_len = max(data.shape[0] for data in data_list)

        labels = pd.Series(label_list, dtype="category")

        self.class_names = labels.cat.categories

        # 将标签转换的编码转化为df
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        return data, labels_df

    def split_by_ratio(self, ratio=0.7):
        # 把DataFrame转换为列表，每个元素是一个小的DataFrame
        groups = list(self.all_df.groupby(by=self.all_df.index))

        # 使用train_test_split随机划分数据集
        train_set, test_set = train_test_split(groups, test_size=1 - ratio, random_state=33)

        test_set, validate_set = train_test_split(test_set, test_size=0.5, random_state=50)

        train_set = pd.concat([group[1] for group in train_set], axis=0)
        test_set = pd.concat([group[1] for group in test_set], axis=0)
        validate_set = pd.concat([group[1] for group in validate_set], axis=0)
        
        return train_set, test_set, validate_set

    """
    def split_by_ratio(self, ratio=0.7):
        data_group = self.all_df.groupby(by=self.all_df.index)

        groups = list(data_group)
        train_groups = groups[:int(len(groups) * ratio)]  # 前70%的组
        test_groups = groups[int(len(groups) * ratio):]  # 后30%的组

        # 将组合并为DataFrame
        train_set = pd.concat([group for name, group in train_groups])
        test_set = pd.concat([group for name, group in test_groups])

        return train_set, test_set
    """

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    # 欧拉角转旋转矩阵
    def euler_to_rot(self, euler):
        r = R.from_euler('zyx', euler, degrees=True)
        rotation_matrix = r.as_matrix()
        return rotation_matrix

    def phone_to_ENU_to_NED_to_car(self, data_phone, phone_quaternion, yaw_car):
        # 步骤1: 从手机坐标系到ENU坐标系
        r_phone = quaternion.as_rotation_matrix(phone_quaternion)
        data_ENU = np.dot(r_phone, data_phone)

        # 步骤2: 从ENU坐标系到NED坐标系
        ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        data_NED = np.dot(ENU_to_NED, data_ENU)

        # 步骤3: 从NED坐标系到车辆坐标系
        rot_NED_to_car = self.euler_to_rot([yaw_car, 180, 0])
        data_car = np.dot(rot_NED_to_car, data_NED)

        return data_car

    def reorientation(self, data):
        print("重定向开始！")

        for index, row in data.iterrows():

            acc_phone = np.matrix(row[['x_acc', 'y_acc', 'z_acc']].values).T

            phone_quaternion = quaternion.quaternion(row['qw'], row['qx'], row['qy'], row['qz'])

            yaw_car = row['bearing'] % 360

            acc_car = self.phone_to_ENU_to_NED_to_car(acc_phone, phone_quaternion, yaw_car)

            acc_car = acc_car.tolist()

            # 更新重定向后的数据
            if len(acc_car) == 3:
                data.loc[index, 'x_acc'] = acc_car[0]
                data.loc[index, 'y_acc'] = acc_car[1]
                data.loc[index, 'z_acc'] = acc_car[2]
            else:
                print(f"Unexpected shape for acc_car_array at index {index}")

        print("重定向结束！")

    def data_KalmanFilter3D(self, data, process_noise=1e-5, observation_noise=1e-4):
        print("卡尔曼滤波开始！")

        # 从DataFrame中提取加速度数据
        acc_reo = data[['x_acc', 'y_acc', 'z_acc']].to_numpy()

        # 初始状态设为零向量 [ax, ay, az]
        initial_state_mean = np.zeros(3)

        # 初始状态协方差设为单位矩阵
        initial_state_covariance = np.eye(3)

        # 状态转移矩阵（对于加速度计数据，使用单位矩阵）
        transition_matrix = np.eye(3)

        # 观测矩阵（单位矩阵，假设所有状态都可以直接观测到）
        observation_matrix = np.eye(3)

        # 过程噪声协方差（可调参数）
        process_noise_covariance = np.eye(3) * process_noise

        # 观测噪声协方差（可调参数）
        observation_noise_covariance = np.eye(3) * observation_noise

        # 初始化卡尔曼滤波器
        kf = KalmanFilter(
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=process_noise_covariance,
            observation_covariance=observation_noise_covariance
        )

        # 应用卡尔曼滤波器到加速度数据
        (filtered_state_means, filtered_state_covariances) = kf.filter(acc_reo)

        # 将过滤后的加速度数据添加到原始DataFrame中
        for i, col in enumerate(['x_acc_clean', 'y_acc_clean', 'z_acc_clean']):
            data[col] = filtered_state_means[:, i]

        print("卡尔曼滤波结束！")

        return data

        # 相同采样率10Hz数据融合

    def merge_data(self, accelerometer, gravity, magnetometer, gps, orientation):

        accelerometer.rename(columns={'x': 'x_acc', 'y': 'y_acc', 'z': 'z_acc'}, inplace=True)

        gravity.rename(columns={'x': 'x_gra', 'y': 'y_gra', 'z': 'z_gra'}, inplace=True)

        magnetometer.rename(columns={'x': 'x_mag', 'y': 'y_mag', 'z': 'z_mag'}, inplace=True)

        merge_data_10_hz = pd.merge_asof(accelerometer, orientation, on='time', suffixes=('_acc', '_ori'),
                                         direction="nearest")

        print('merge accelerometer and orientation,mergeData shape: ', merge_data_10_hz.shape)

        merge_data_10_hz = pd.merge_asof(merge_data_10_hz, gravity, on='time', suffixes=('_gra', '_merge'),
                                         direction="nearest")

        print('merge gravity and mergeData,mergeData shape: ', merge_data_10_hz.shape)

        merge_data_10_hz = pd.merge_asof(merge_data_10_hz, magnetometer, on='time', suffixes=('_mag', '_merge'),
                                         direction="nearest")

        print('merge magnetometer and mergeData,mergeData shape: ', merge_data_10_hz.shape)

        # 融合不同采样率数据

        gps_10_hz = pd.DataFrame(columns=gps.columns)

        gps_10_hz['time'] = merge_data_10_hz['time']

        for column in gps.columns:
            if column == 'time':
                continue
            else:
                gps_10_hz[column] = np.interp(gps_10_hz['time'], gps['time'], gps[column])

        merge_data_10_hz = pd.merge(merge_data_10_hz, gps_10_hz, how='left', on='time', suffixes=('_merge', '_GPS'))

        print('merge GPS and mergeData,mergeData shape: ', merge_data_10_hz.shape)

        return merge_data_10_hz

    def acc_clean_figure(self, accelerometer_clean):
        signal = go.Figure()

        accelerometer_clean.index = pd.to_datetime(accelerometer_clean['time'], origin='1970-01-01 08:00:00',
                                                   unit='ns')

        # signal.add_trace(go.Scatter(x = GPS_10Hz.index , y = GPS_10Hz['speed'], name = 'speed', mode = 'markers'))

        signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['x_acc'], name='x'))
        signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['y_acc'], name='y'))
        signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['z_acc'], name='z'))
        signal.add_trace(
            go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['x_acc_clean'], name='x_clean'))
        signal.add_trace(
            go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['y_acc_clean'], name='y_clean'))
        signal.add_trace(
            go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['z_acc_clean'], name='z_clean'))

        signal.show()

    def read_with_template(self, filename, template):
        df = pd.read_csv(filename, usecols=template)
        return df


"""
    def pad_data_to_max_rows(self, dataList):

        # 根据dataList中具有最大行数的数据，对其他数据进行零填充。

        # 参数:
           # dataList (list of np.array): 待填充的数据列表。

        #  padded_dataList (list of np.array): 填充后的数据列表。


        # 2. 如果某个数据的行数小于最大行数，使用零填充它
        padded_dataList = []
        for data in dataList:
            if data.shape[0] < max_rows:
                # 使用零进行填充
                padding = np.zeros((max_rows - data.shape[0], data.shape[1]), dtype=data.dtype)
                padded_data = np.vstack((data, padding))
            else:
                padded_data = data
            padded_dataList.append(padded_data)

        return padded_dataList
"""

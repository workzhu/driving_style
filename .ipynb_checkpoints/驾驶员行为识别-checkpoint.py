import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import quaternion
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.transform import Rotation as R
from pykalman import KalmanFilter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

base_dir = 'D:\实验数据'  # 数据集的路径

# 准备一个列表来收集数据和标签
data_list = []
label_list = []


# 重定向


# 欧拉角转旋转矩阵
def euler_to_rot(euler):
    r = R.from_euler('zyx', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def phone_to_ENU_to_NED_to_car(data_phone, phone_quaternion, yaw_car):
    # 步骤1: 从手机坐标系到ENU坐标系
    r_phone = quaternion.as_rotation_matrix(phone_quaternion)
    data_ENU = np.dot(r_phone, data_phone)

    # 步骤2: 从ENU坐标系到NED坐标系
    ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    data_NED = np.dot(ENU_to_NED, data_ENU)

    # 步骤3: 从NED坐标系到车辆坐标系
    rot_NED_to_car = euler_to_rot([yaw_car, 180, 0])
    data_car = np.dot(rot_NED_to_car, data_NED)

    return data_car


def reorientation(data):
    print("重定向开始！")

    for index, row in data.iterrows():

        acc_phone = np.matrix(row[['x_acc', 'y_acc', 'z_acc']].values).T

        phone_quaternion = quaternion.quaternion(row['qw'], row['qx'], row['qy'], row['qz'])

        yaw_car = row['bearing'] % 360

        acc_car = phone_to_ENU_to_NED_to_car(acc_phone, phone_quaternion, yaw_car)

        acc_car = acc_car.tolist()

        # 更新重定向后的数据
        if len(acc_car) == 3:
            data.loc[index, 'x_acc'] = acc_car[0]
            data.loc[index, 'y_acc'] = acc_car[1]
            data.loc[index, 'z_acc'] = acc_car[2]
        else:
            print(f"Unexpected shape for acc_car_array at index {index}")

    print("重定向结束！")


def data_KalmanFilter3D(data, process_noise=1e-5, observation_noise=1e-4):
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

def merge_data(accelerometer, gravity, magnetometer, gps, orientation):
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


def acc_clean_figure(accelerometer_clean):
    signal = go.Figure()

    accelerometer_clean.index = pd.to_datetime(accelerometer_clean['time'], origin='1970-01-01 08:00:00', unit='ns')

    # signal.add_trace(go.Scatter(x = GPS_10Hz.index , y = GPS_10Hz['speed'], name = 'speed', mode = 'markers'))

    signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['x_acc'], name='x'))
    signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['y_acc'], name='y'))
    signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['z_acc'], name='z'))
    signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['x_acc_clean'], name='x_clean'))
    signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['y_acc_clean'], name='y_clean'))
    signal.add_trace(go.Scatter(x=accelerometer_clean.index, y=accelerometer_clean['z_acc_clean'], name='z_clean'))

    signal.show()


# main

dataList = []
labelList = []


def read_with_template(filename, template):
    df = pd.read_csv(filename, usecols=template)
    return df


def pad_data_to_max_rows(dataList):
    """
    根据dataList中具有最大行数的数据，对其他数据进行零填充。

    参数:
        dataList (list of np.array): 待填充的数据列表。

    返回:
        padded_dataList (list of np.array): 填充后的数据列表。
    """

    # 1. 确定最大行数
    max_rows = max(data.shape[0] for data in dataList)

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


def data_loader(base_dir):
    # 遍历所有驾驶员
    for driver in os.listdir(base_dir):
        driver_dir = os.path.join(base_dir, driver)

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
                accelerometer = read_with_template(os.path.join(trip_dir, 'Accelerometer.csv'), ['time', 'seconds_elapsed', 'x', 'y', 'z'])

                orientation = read_with_template(os.path.join(trip_dir, 'Orientation.csv'), ['time', 'qw', 'qx', 'qy', 'qz', 'roll', 'pitch', 'yaw'])

                gravity = read_with_template(os.path.join(trip_dir, 'Gravity.csv'), ['time', 'x', 'y', 'z'])

                magnetometer = read_with_template(os.path.join(trip_dir, 'Magnetometer.csv'), ['time', 'x', 'y', 'z'])

                gps = read_with_template(os.path.join(trip_dir, 'Location.csv'), ['time', 'bearing', 'bearingAccuracy', 'speed',
                                                                                     'speedAccuracy', 'latitude', 'longitude'])

                temp_data = merge_data(accelerometer, gravity, magnetometer, gps, orientation)

                # 重定向
                reorientation(temp_data)

                # 对加速度进行卡尔曼滤波
                data_KalmanFilter3D(temp_data, 1e-6, 1e-5)

                # 保存清洗后的数据
                temp_data.to_csv(os.path.join(trip_dir, 'clean_data.csv'), index=False)

            acc_clean_figure(temp_data)

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

            dataList.append(temp_data.values)

            # 提取行程标签
            trip_label = trip.split('_')[2]  # 假设标签是文件夹名称的第三个元素，且normal为0，abnormal为1

            # 将数据和标签添加到列表中
            # data_list.append(scaled_data)
            labelList.append(0 if trip_label == 'normal' else 1)

            print("**********************************************************")

    print("数据读取结束！共读取", len(dataList), "次行程。")

    return dataList, labelList


dataList, labelList = data_loader(base_dir)



# 使用这个函数对dataList进行填充
padded_dataList = pad_data_to_max_rows(dataList)

# 检查填充后的dataList中第一个元素的形状和数据类型
print(padded_dataList[0].shape, padded_dataList[0].dtype)

# 将填充后的数据列表转换为一个 numpy.ndarray
padded_dataList = np.array(padded_dataList)



# 将数据和标签转换为张量
data_tensor = torch.tensor(padded_dataList, dtype=torch.float32)
label_tensor = torch.tensor(labelList, dtype=torch.long)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_tensor.numpy(), label_tensor.numpy(), test_size=0.2, random_state=42)


# 假设 X_train 和 X_test 是三维的，形状为 [样本数, 时间步数, 特征数]
num_samples, num_timesteps, num_features = X_train.shape

# 重塑为二维，进行归一化
X_train_reshaped = X_train.reshape(-1, num_features)  # 将时间步展平
X_test_reshaped = X_test.reshape(-1, num_features)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# 将归一化后的数据重塑回原来的三维形状
X_train_scaled = X_train_scaled.reshape(num_samples, num_timesteps, num_features)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# 转换回张量
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建训练和测试数据集
train_dataset = TensorDataset(X_train_scaled, y_train)
test_dataset = TensorDataset(X_test_scaled, y_test)

# 创建训练和测试数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# 检查GPU是否可用，并相应地设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, 128)

        # 定义全连接层
        self.fc = nn.Linear(128, 64)

        # 定义全连接层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 通过LSTM层
        out, _ = self.lstm(x)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层
        out = self.fc(out)

        return out



# 实例化模型
model = LSTMModel(input_size=25, hidden_size=64, num_layers=4, num_classes=2)

model.to(device)  # 将模型移至设备（GPU或CPU）


# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

average_losses = []  # 记录每次迭代的平均损失

# 训练模型
for epoch in range(100):

    epoch_losses = []  # 用于记录这个迭代中所有批次的损失

    for data_batch, label_batch in train_loader:
        # 将数据和标签移至设备（GPU或CPU）

        data_batch = data_batch.to(device)
        label_batch = label_batch.to(device)


        # 前向传播
        output = model(data_batch)

        # 计算损失
        loss = loss_fn(output, label_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    average_loss_for_epoch = sum(epoch_losses) / len(epoch_losses)
    average_losses.append(average_loss_for_epoch)

    print(f"Epoch {epoch} finished training.")

# 绘制损失值图表
plt.plot(average_losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Training Loss Over Epochs')
plt.show()


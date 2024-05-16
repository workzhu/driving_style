
from data_provider.MyDataLoader import MyDataLoader
from torch.utils.data import DataLoader, random_split

import torch

def data_provider(args):
    batch_size = args.batch_size  # bsz for train and valid

    # 读数据（对此类进行改写）
    dataset = MyDataLoader(
        root_path=args.root_path,
        window_size=args.window_size,
        step_size=args.step_size
    )

    # 计算各个子集的大小
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size

    print('total_size:', total_size)
    print('train_size:', train_size)
    print('valid_size:', valid_size)
    print('test_size:', test_size)

    # 划分数据集
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset, train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


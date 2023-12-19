from data_provider.data_loader import UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.MyDataLoader import MyDataLoader
from data_provider.uea import subsample, interpolate_missing, Normalizer
import copy


def make_dataset(data_set, df, limit_size=None):
    temp_set = copy.deepcopy(data_set)
    temp_set.all_df = df
    temp_set.all_IDs = df.index.unique()
    
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
    # 标准化
    normalizer = Normalizer()

    temp_set.feature_df = normalizer.normalize(temp_set.feature_df)

    return temp_set


def data_provider(args):
    batch_size = args.batch_size  # bsz for train and valid
    drop_last = True

    # 读数据（对此类进行改写）
    data_set = MyDataLoader(
        root_path=args.root_path
    )

    train_df, test_df, vali_df = data_set.split_by_ratio(ratio=0.7)

    train_set = make_dataset(data_set, train_df)
    test_set = make_dataset(data_set, test_df)
    vali_set = make_dataset(data_set, vali_df)
    
    print("训练集行程ID：", list(train_set.all_IDs))
    print("验证集行程ID：", list(vali_set.all_IDs))
    print("测试集行程ID：", list(test_set.all_IDs))
    
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
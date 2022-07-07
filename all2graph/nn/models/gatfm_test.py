import os
import platform
import string
import json
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

import all2graph as ag

if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    dst_dir_path = 'train_data'
    if os.path.exists(dst_dir_path):
        shutil.rmtree(dst_dir_path)

    targets = ['y', 'z']
    train_data_df = []
    for i in ag.tqdm(range(100)):
        x = []
        for _ in range(np.random.randint(1, 200)):
            item = {
                k: list(np.random.choice(list(string.ascii_letters)+list(string.digits), size=np.random.randint(1, 10)))
                for k in np.random.choice(list(string.ascii_letters), size=np.random.randint(1, 10))
            }
            x.append(item)
        x = json.dumps(x, indent=None, separators=(',', ':'))
        train_data_df.append({'x': x, np.random.choice(targets): np.mean(list(map(ord, x)))})

    train_data_df = pd.DataFrame(train_data_df)
    for target in targets:
        train_data_df[target] = (train_data_df[target] - train_data_df[target].mean()) / train_data_df[target].std()
    train_data_df['time'] = None

    train_path_df = ag.split_csv(
        src=train_data_df, # 原始数据，可以是dataframe，path和list of path
        dst=dst_dir_path, # 分片后的文件夹地址
        chunksize=10, # 每一片的大小，建议根据你的机器配置设置，设的太大会影响后续训练速度，增大内存开销，设的太小会产生大量小文件
        drop_cols=targets
    )

    data_parser = ag.JsonParser(
        # 必填，包含json数据的列
        json_col='x',
        # 必填，包含样本观察时间戳的列
        time_col='time',
        # 如果time_col的值不为空，按么必填，需要是时间戳的格式，如”%Y-%m-%d“
        time_format=None,
        # 标签名
        targets=targets
    )

    model = ag.nn.GATFM(
        d_model=8,
        num_layers=6,
        num_heads=2,
        data_parser=data_parser,
        check_point='check_point',
        num_featmaps=2,
        to_bidirected=True,
        mask_prob=0.1,
        mask_loss_weight=0.1
    )
    if torch.cuda.is_available():
        model.cuda()

    trainer = model.fit(
        train_path_df,
        epoches=2,  # 训练轮数
        batch_size=16,  # 梯度下降的样本数量
        chunksize=100,  # 分析阶段的分片样本数量
        loss=ag.nn.DictLoss(torch.nn.MSELoss()),  # 损失函数
        metrics={'mse': ag.Metric(mean_squared_error, label_first=True)},  # 评估函数
        valid_data=[train_path_df],
        processes=os.cpu_count()-1,   # 多进程数量
    )

    with torch.no_grad():
        model.eval()
        print(model(train_data_df.iloc[:10]))

    print(model.predict('train_data', chunksize=16, embedding=True))
    print(model.predict('train_data', chunksize=16))

    shutil.rmtree(dst_dir_path)
    shutil.rmtree(model.check_point+'.'+model.version)
    os.remove(dst_dir_path+'_path.zip')

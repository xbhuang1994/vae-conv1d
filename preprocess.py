# -*- encoding: utf-8 -*-
'''
File    :   preprocess.py
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   数据切分和预处理
'''

from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import numpy as np
import glob
import pandas as pd
from datetime import datetime
import time
from time import mktime
import pytz
import random
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category=Warning)

def preprocess_data(data):
    data = pd.DataFrame(data, columns=['time', "price", "vol"])
    data['times'] = data['time'].apply(lambda x: datetime.fromtimestamp(mktime(time.localtime(x/1000)), tz=pytz.utc))
    data['price'] = data['price'].apply(lambda x: x/10000)
    data = data.set_index("times")
    data1 = data["1970-01-01  09:30":"1970-01-01  11:30"]
    data2 = data["1970-01-01  13:00":]
    data1['time_sub'] = data1['time'] - data1['time'].shift(1)
    data1.loc[0:1, ("time_sub")] = 0
    data2['time_sub'] = data2['time'] - data2['time'].shift(1)
    # 中午时间拉大间隔
    data2.loc[0:1, ("time_sub")] = 2*60*1000
    data = pd.concat([data1, data2])
    data = data[['price', "time_sub", "vol"]]
    data = data.dropna()
    return data

# 按照多个数据进行切片
# batch_sample = 100  
def text_preprocess(filename,batch_sample = 100,step = 50):
    with open(filename, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        data = data['arr_0']
    if len(data) == 0:
        return []
    buy_data = preprocess_data(data)
    new_data = buy_data
    new_data = new_data.sort_index()
    number = len(new_data)
    all_data = []
    # buy，sell个12个数据
    for i in range(0, number-batch_sample, step):
        sub_data = new_data[i:i+batch_sample]
        # 不对价格的方差进行归一化
        sub_data = sub_data.apply(lambda x: (x-np.min(x))/max((np.max(x)-np.min(x)), 1))
        sub_data = sub_data.values
        all_data.append(sub_data)
    return all_data


if __name__ == "__main__":
    pool = ProcessPoolExecutor(28)
    tasks = []
    for filename in tqdm(glob.glob("/data/trade/2022*.npz")):
        task = pool.submit(text_preprocess, filename)
        tasks.append(task)
    all_data = []
    for task in tqdm(as_completed(tasks)):
        all_data.extend(task.result())

    random.shuffle(all_data)
    number = len(all_data)
    ratio = 0.9
    train_data = all_data[:int(number*ratio)]
    val_data = all_data[int(number*ratio):]
    np.save("data/train_data.npy", train_data)
    np.save("data/val_data.npy", val_data)

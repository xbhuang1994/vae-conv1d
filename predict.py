# -*- encoding: utf-8 -*-
'''
File    :   predict.py
Time    :   2022/09/25 16:16:53
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   加载模型和文件并预测
'''


import torch
from preprocess import text_preprocess
from tqdm import tqdm
import random
import pytz
from time import mktime
import time
from datetime import datetime
import pandas as pd
import glob
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/cat_vae_stock.yaml')

parser.add_argument('--checkpoint',
                    help='定义需要加载的模型文件',
                    default='logs/CategoricalVAEStock/version_6/checkpoints/epoch=0-step=7462.ckpt')


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])
# experiment = VAEXperiment(model, config['exp_params'])
experiment = VAEXperiment.load_from_checkpoint(args.checkpoint, vae_model=model, params=config['exp_params'])
model = experiment.model
# 保存模型文件
torch.save(model, "logs/CategoricalVAEStock/version_6/checkpoints/model.pt")
model = model.cuda()
model.eval()
datas = text_preprocess("/data/trade/20220701_110044.SZ_buy.npz")
batch_size = 5
number = len(datas)

for i in range(0, number, batch_size):
    batch_data = datas[i:i+number]
    sub_data = [np.transpose(sub_data, (1, 0)).astype(np.float32) for sub_data in batch_data]
    sub_data = np.stack(sub_data)
    sub_data = torch.tensor(sub_data)
    sub_data = sub_data.cuda()
    output = model.encode(sub_data)[0]
    
    # print("output:",output[-2:])
    output = output.mean(1)
    predicts = torch.softmax(output, -1)
    labels = predicts.argmax(-1).cpu().numpy()
    print(labels)

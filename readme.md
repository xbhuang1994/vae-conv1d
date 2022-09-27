首先将数据解压放在当前项目下的trade文件夹中

然后运行： python  preprocess.py文件，处理数据和切分训练和验证集

然后训练模型： python run.py

最后进行预测，根据实际需要加载训练后的模型，已经相应的数据，执行 python predict.py。

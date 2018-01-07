
* 环境：python3.6, pytorch(gpu), scipy, sklearn
* 文件：
  * dataloader.py 数据加载，包括数据集的划分、数据的过滤和降维
  * main.py 训练及验证
  * model.py 模型
  * test.py 测试代码的文件，可以忽略
* 数据集
  * 文件树：data: {datas: [sleep_data_row3_*.mat, train_1222.mat, test_1222.mat], labels: [HypnogramAASM_subject*.txt]}
  * 其中train_1222.mat和test_1222.mat分别为新增的训练数据和测试数据
* 需要调整的：model.py中的模型，main.py中的训练速率，dataloader.py中sample函数（即降维方法）
* 目前模型为全连接层和激活层构成的网络，在验证集上的精度达到了74.30%

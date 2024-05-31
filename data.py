# 数据集下载的库
from torchvision.datasets import MNIST
 
# # 将MNIST数据可视化的库
# import matplotlib.pyplot as plt
 
# 数据转换的方式
from torchvision import transforms

# 下载数据集
train_dataset_no = MNIST('./data/with_notrans',train=True,download=True)
test_dataset_no = MNIST('./data/with_notrans',train=False,download=True)
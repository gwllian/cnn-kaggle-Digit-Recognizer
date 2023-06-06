
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
device = torch.device('cuda')
from torch.utils.data import Dataset, DataLoader, random_split

train = pd.read_csv('./train.csv')
labels=torch.from_numpy(np.array(train['label']))
train.pop('label')#抛弃标签列
ones = torch.sparse.torch.eye(10)
#根据指定索引和维度保留单位矩阵中的一条数据即为one-hot编码
label_one_hot=ones.index_select(0,labels)

#还原图像，用卷积
imgs=torch.from_numpy(np.array(train))
imgs = imgs.to(torch.float32)
imgs=imgs.view(42000,28,28)

#归一化
mean = torch.mean(imgs, dim=0)
std = torch.std(imgs, dim=0)
imgf= torch.div(torch.sub(imgs, mean), std)
nan_mask = torch.isnan(imgf)
imgf= torch.where(nan_mask, torch.zeros_like(imgf), imgf)
imgf=torch.unsqueeze(imgf,1)


#超参数设置
learning_rate=1e-3
epoch=5
#模型
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.conv1=nn.Conv2d(1,3,kernel_size=3,padding=1,stride=1)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(3,16,kernel_size=3,padding=1,stride=1)
        self.pool2=nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.lin1=nn.Linear(28*28,10)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))  # 进行第一次卷积、ReLU激活和最大池化
        x = self.pool2(F.relu(self.conv2(x)))  # 进行第二次卷积、ReLU激活和最大池化
        x = self.flatten(x)  # 将特征图展平为一维向量
        x = self.lin1(x)  # 进行输出层
        return x


net=model()
net.to(device)
# 损失函数
loss = nn.CrossEntropyLoss()


# 优化器梯度下降
optim = torch.optim.SGD(params=net.parameters(), lr=learning_rate)


#定义数据集

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        # 从数据和标签中获取对应索引的数据和标签，并返回
        data_item = self.data[index]
        label_item = self.labels[index]
        return data_item, label_item

    def __len__(self):
        return len(self.data)


dataset = MyDataset(imgf, label_one_hot)
train_size = int(0.8 * len(dataset))  # 训练集大小为数据集大小的 80%
test_size = len(dataset) - train_size  # 测试集大小为数据集大小的 20%
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
testlegth=len(test_dataset)
trainleg=len(train_dataset)

step=0
for i in range(0,epoch):
    net.train()#表示模型开始训练模式，更新参数
    true_num=0
    for data in train_loader:
        images, label = data
        #移到gpu
        images = images.to(device)
        label = label.to(device)
        outputs=net(images)

        loss_results = loss(outputs, label.argmax(1))

        optim.zero_grad()#optimizer.zero_grad()# 是 PyTorch 中定义优化器的一个方法，它会将模型中所有可训练的参数的梯度清零。在训练神经网络时，通常需要在每次迭代之前调用这个函数。因为如果不清零梯度，那么优化器在更新权重时会累加之前的梯度。

        loss_results.backward()#optim.step()是PyTorch的一个方法，它根据反向传播过程中计算的梯度来更新优化器的参数。它通常在 loss.backward() 之后被调用，以更新模型的权重。
        optim.step()
        true_num += (outputs.argmax(1) == label.argmax(1)).sum()
        step += 1
        if(step%500==0):
            print("当前训练轮数：{} 训练损失:{}".format(i,loss_results.item()))

    accuracy = true_num / trainleg
    print("<*>After this round, {}% accuracy in Train_dataset.".format(accuracy * 100))
    #测试
    net.eval()#停止参数修改
    true=0
    running_loss=0
    with torch.no_grad():
        for data in test_loader:
            images, label=data
            images = images.to(device)
            label = label.to(device)

            outputs = net(images)
            results = (outputs.argmax(1) == label.argmax(1)).sum()
            true += results
            loss_results = loss(outputs, label)
            running_loss += loss_results.item()
    accuracy = true/testlegth

    print("经过这轮，测试集准确度为：{}%".format(accuracy*100))
    print("经过这轮，测试集损失为：{}".format(running_loss))
    #可视化

    torch.save(net, "./Net_save/Net_save_{}.pth".format(i))
    print("Net save successfully.")


import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split



df_test = pd.read_csv('./test.csv')
test_imgs=torch.from_numpy(np.array(df_test))
test_imgs = test_imgs.to(torch.float32)
device = torch.device('cuda')
#测试集预测部分
model=torch.load(r'C:\Users\yang\PycharmProjects\pythonProject4\Net_save\Net_save_19.pth')
#传入GPU
test_imgs=test_imgs.to(device)
#计算结果
pre=model(test_imgs).max(1)
#将结果转为提交kaggle的格式
res={}
pre = pre.cpu().numpy()
pre_size=pre.shape[0]
num = [i for i in range(1,pre_size+1)]
res_df=pd.DataFrame({
    'ImageId':num,
    'Label':pre
})

#d导出为CSV文件
res_df.to_csv('res.csv',index=False)


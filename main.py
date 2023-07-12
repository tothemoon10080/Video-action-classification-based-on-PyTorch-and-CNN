#导入实验所需要的第三方库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import timeit
from PIL import Image
import os
import numpy as np
import scipy.io
import torchvision.models.inception as inception
from multiprocessing import freeze_support


#数据加载
label_mat=scipy.io.loadmat('./datasets/q3_2_data.mat')#读取数据集
label_train=label_mat['trLb']#训练集标签
label_val=label_mat['valLb']#验证集标签


#Dataset类
class ActionDataset(Dataset):#定义数据集类
    """Action dataset."""

    def __init__(self,  root_dir, labels=[], transform=None):
        """
        Args:
            root_dir (string): 整个数据的路径。
            labels(list): 图片的标签。
            transform (callable, optional): 想要对数据进行的处理函数。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length=len(os.listdir(self.root_dir))
        self.labels=labels
    
    def __len__(self):      # 该方法只需返回数据的数量。
        return self.length*3    # 因为每个视频片段都包含3帧。

    def __getitem__(self, idx):     # 该方法需要返回一个数据。
        
        folder=idx//3+1
        imidx=idx%3+1
        folder=format(folder,'05d')
        imgname=str(imidx)+'.jpg'
        img_path = os.path.join(self.root_dir,folder,imgname)
        image = Image.open(img_path)

        if len(self.labels)!=0:
            Label=self.labels[idx//3][0]-1
        if self.transform:      # 如果要先对数据进行预处理，则经过transform函数。
            image = self.transform(image)
        if len(self.labels)!=0:
            sample={'image':image,'img_path':img_path,'Label':Label}
        else:
            sample={'image':image,'img_path':img_path}
        return sample

#Dataloader类
image_dataset_train=ActionDataset(root_dir='./datasets/trainClips/',
                                  labels=label_train,transform=T.ToTensor())#训练集
image_dataloader_train = DataLoader(image_dataset_train, batch_size=32,
                                    shuffle=True, num_workers=4)#训练集加载器
image_dataset_val=ActionDataset(root_dir='./datasets/valClips/',
                                labels=label_val,transform=T.ToTensor())#验证集
image_dataloader_val = DataLoader(image_dataset_val, batch_size=32,
                                  shuffle=False, num_workers=4)#验证集加载器
image_dataset_test=ActionDataset(root_dir='./datasets/testClips/',
                                 labels=[],transform=T.ToTensor())#测试集
image_dataloader_test = DataLoader(image_dataset_test, batch_size=32,
                                   shuffle=False, num_workers=4)#测试集加载器
dtype = torch.FloatTensor # 这是pytorch所支持的cpu数据类型中的浮点数类型。
print_every = 100   # 这个参数用于控制loss的打印频率，因为我们需要在训练过程中不断的对loss进行检测。
def reset(m):   # 这是模型参数的初始化
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()#初始化模型参数
        
#数据解释和处理
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # 读取各个维度。 
        return x.view(N, -1)  # -1代表除了特殊声明过的以外的全部维度。

fixed_model_base = nn.Sequential( # 这里我们使用了一个简单的卷积神经网络。
                nn.Conv2d(3, 8, kernel_size=7, stride=1), #3*64*64 -> 8*58*58
                nn.ReLU(inplace=True),#inplace=True表示直接对输入进行覆盖计算，不会占用额外的内存。
                nn.MaxPool2d(2, stride = 2),    # 8*58*58 -> 8*29*29
                nn.Conv2d(8, 16, kernel_size=7, stride=1), # 8*29*29 -> 16*23*23
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2), # 16*23*23 -> 16*11*11
                Flatten(),# 16*11*11 -> 1936
                nn.ReLU(inplace=True),
                nn.Linear(1936, 10)     # 1936 = 16*11*11
            )

fixed_model = fixed_model_base.type(dtype)#将模型的参数类型转换为cpu的Float类型。

x = torch.randn(32, 3, 64, 64).type(dtype)#随机生成一个输入数据。
x_var = Variable(x.type(dtype)) # 需要将其封装为Variable类型。
ans = fixed_model(x_var)        # 将数据输入模型，得到输出。

np.array_equal(np.array(ans.size()), np.array([32, 10]))   # 检查模型输出的维度是否正确。

def train(model, loss_fn, optimizer, dataloader, num_epochs):#训练函数
    for epoch in range(num_epochs):#循环训练
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))#打印当前训练轮数

        check_accuracy(fixed_model, image_dataloader_val)#每轮训练结束后，都对验证集进行一次测试。
        
        model.train() # 模型的.train()方法切换进入训练模式，这会启用dropout和batch normalization。
        for t, sample in enumerate(dataloader):#循环每个batch
            x_var = Variable(sample['image'])   # 取得一个batch的图像数据。
            y_var = Variable(sample['Label'].long()) # 取得对应的标签。

            scores = model(x_var)   # 得到输出。
            
            loss = loss_fn(scores, y_var)   # 计算loss。
            if (t + 1) % print_every == 0:  # 每隔一段时间打印一次loss。
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))#打印loss

            # 三步更新参数。
            optimizer.zero_grad()#清空梯度
            loss.backward()#反向传播
            optimizer.step()#更新参数

def check_accuracy(model, loader):#测试函数

    num_correct = 0
    num_samples = 0

    model.eval() # 模型的.eval()方法切换进入评测模式，对应的dropout等部分将停止工作。
    for t, sample in enumerate(loader):
        x_var = Variable(sample['image'])
        y_var = sample['Label']
       
        scores = model(x_var)
        _, preds = scores.data.max(1) # 找到可能最高的标签作为输出。

        num_correct += (preds.numpy() == y_var.numpy()).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

optimizer = torch.optim.RMSprop(fixed_model_base.parameters(), lr = 0.0001)#优化器

loss_fn = nn.CrossEntropyLoss()#损失函数

torch.random.manual_seed(54321)#设置随机种子
fixed_model.cpu()#将模型的参数类型转换为cpu的Float类型。
fixed_model.apply(reset) # 重新初始化模型参数。
fixed_model.train() # 切换进入训练模式。
if __name__ == '__main__':#主函数
    freeze_support()#多进程
    train(fixed_model, loss_fn, optimizer,image_dataloader_train, num_epochs=5) # 训练模型。
    check_accuracy(fixed_model, image_dataloader_val)#测试模型。
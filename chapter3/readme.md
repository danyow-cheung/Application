# Developing a Powerful Deep Learning Model
主題概念主要有
- Going through the basic theory of DL
- Understanding the components of DL frameworks
- Implementing and training a model in Pytorch
- Implementing and training a model in TF
- Decomposing a complex,state-of-the-art model implementation

## Going through the basic theory of DL

### How does DL work?
人工神经网络通常包含幾個神經元的組合。包含權重和偏差


DL 中的操作基于数值。 所以，网络的输入数据必须转换为数值。 例如，红色、绿色、
蓝色 (RGB) 颜色代码是使用数值表示图像的标准方式。 在
对于文本数据，经常使用词嵌入。 

类似地，网络的输出将是一组数值。 这些值的解释可能因任务和定义而异。

### DL model training
总的来说，训练 ANN 是一个寻找一组权重、偏差和激活函数的过程，这些权重、偏差和激活函数使网络能够从数据中提取有意义的模式


## Components of DL frameworks
由于无论底层任务如何，模型训练的配置都遵循相同的过程，
许多工程师和研究人员已将常见的构建块组合成框架。
大多数框架通过保持数据加载逻辑和模型定义独立于训练逻辑来简化深度学习模型开发。
### The data loading logic 
数据加载逻辑包括从将原始数据加载到内存中到准备每个样本以进行训练和评估的所有内容。 在许多情况下，训练集、验证集和测试集的数据存储在不同的位置，因此每个数据集都需要不同的加载和准备逻辑。


标准框架将这些逻辑与其他构建块分开，以便可以使用不同的数据集以动态方式训练模型，而模型方面的更改最少。 此外，框架还标准化了这些逻辑的定义方式，以提高可重用性和可读性

### The model definition
另一个构建块，即模型定义，指的是 ANN 架构本身以及相应的前向和后向传播逻辑。 尽管使用算术运算构建模型是一种选择，但标准框架提供了通用层定义，用户可以将这些定义组合在一起以构建复杂的模型。

因此，用户负责实例化必要的网络组件、连接组件并定义模型应如何进行训练和推理。

### Model training logic 
根据学习任务的类型，损失函数可以分为两大类： 分类损失和回归损失

## Implementing and training a model in Pytorch
### pytorch data loading logic 
当 Dataset 类处理获取单个样本时，模型训练会接收输入批量数据，需要重新洗牌以减少模型过度拟合。 DataLoader 通过提供简单的 API 为用户消除了这种复杂性。 此外，它在幕后利用 Python 的多处理功能来加速数据检索。
```python
from torch.utils.data import Dataset 
class SampleDataset(Dataset):
    def __len__(self):
        "return the number of samples"
    def __getitem(self,index):
        '''load and returns a sample from the dataset at the given index'''
```

PL 的 LightningDataModule 封装了处理数据所需的所有步骤。 关键组件包括下载和清理数据、预处理每个样本以及将每种类型的数据集包装在 DataLoader 中。 以下代码片段描述了如何创建 LightningDataModule 类。

该类具有用于下载和预处理数据的prepare_data函数，以及用于为每种类型的数据集实例化DataLoader的三个函数：train_dataloader、val_dataloader和test_dataloader：
```python
from torch.utils.data  import DataLoader
from pytorch_lightning.core.lighting import LightningDataModule

class SampleDataModule(LightningDataModule):
    def prepare_data(self):
        '''
        download and preprocess the data: triggered only on single gpu
        
        '''
    def setup(self):
        '''
        define necessary components for data loading on each gpu
        '''
    def train_dataloader(self):
        '''
        define train data loader
        '''
        return data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle=True
        )
    def val_dataloader(self):
        '''define validation data loader'''
        return data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        '''define test data loader'''
        return data.DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle=False
        )
    
```

### PyTorch model definition
PL 的主要优势来自 LightningModule，它简化了复杂的组织
PyTorch 代码分为六个部分：
- Computation
- The train loop
- The validation loop
- The test loop
- The prediction loop
- Optimizers and LR scheduler

模型架构是计算部分的一部分。 必要的层被实例化
在__init__方法内部，计算逻辑在forward方法中定义。
在下面的代码片段中，三个线性层注册到LightningModule
module内部的__init__方法，里面定义了它们之间的关系
前向方法：
```python
from pytorch_lightning import LightningModule
from torch import nn 
class SampleModel(LightningModule):
    def __init__(self):
        '''
        instantiate necessary layers
        '''
        self.individual_layer_1 = nn.Linear(...,...)
        self.individual_layer_2 = nn.Linear(...,...)
        self.individual_layer_3 = nn.Linear(...,...)
    def forward(self,input):
        '''
        define forward propagation logic
        '''
        output_1 = self.individual_layer_1(input)
        output_2 = self.individual_layer_2(output_1)
        final_output = self.individual_layer_3(output_2)
        return final_output
```
定义网络的另一种方法是使用 torch.nn.Sequential，如以下代码所示。 使用此模块，可以将一组层分组在一起，并自动实现输出链接：
```python
class SampleModel(LightningModule):
    def __init__(self):
        '''
        instantiate necessary layers
        '''
        self.multiple_layers =  nn.Sequential(
            nn.Linear(,),
            nn.Linear(,),
            nn.Linear(,),

        )
    def forward(self,input):
        '''define forward propagation logic'''
        final_output = self.multiple_layers(input)
        return final_output
```
#### Pytorch DL layers
深度学习框架的主要好处之一来自于各种层定义：梯度计算
逻辑已经是层定义的一部分，因此您可以专注于寻找最佳模型架构
##### Pytorch dense(linear) layers 
```python
linear_layer = torch.nn.Linear(
    in_features,out_features
)
input_tensor = torch.rand(N,*,in_features)
output_tensor = linear_layer(input_tensor)
```
##### PyTorch pooling layers
```python
# 2d max pooling 
max_pool_layer = torch.nn.MaxPool2d(
    kernel_size,stride = None,
    padding = 0,dilation =1 
)
input_tensor = torch.rand(N,C,H,W)
output_tensor = max_pool_layer(input_tensor)
```
還有另外一種方法
```python
# 2d average pooling 
avg_pool_layer = torch.nn.AvgPool2d(kernel_size,stride=None,padding=0)
input_tensor = torch.rand(N,C,H,W)
output_tensor = avg_pool_layer(input_tensor)
```
##### PyTorch normalization layers
標準化通常用於資料處理，其目的是在不扭曲分佈的情况下將數值數據縮放到通用尺度。 在DL的情况下，使用歸一化層來訓練具有更大數值穩定性的網絡

最流行的規範化層是批次處理規範化層，它縮放一組值
在小批量中。 在下麵的程式碼片段中，我們介紹了torch.nn.BatchNorm2d，這是一個為具有額外通道維度的2D張量的小批量設計的批量規範化層：

```python
batch_norm_layer = torch.nn.BatchNorm2d(
    num_features,
    eps =1e-05,
    momentum=0.1,
    affine=True,
)
input_tensor = torch.rand(N,C,H,W)
output_tensor = batch_norm_layer(input_tensor)
```
在各種參數中，您應該注意的主要參數是num_features，
其訓示通道的數量。 層的輸入是4D張量，其中每個索引
表示批次大小（N）、通道數（C）、影像高度（H）和
影像（W）。

##### Pytorch dropout layer 
```python
dropout_layer = torch.nn.Dropout2d(
    p = 0.5
)
input_tensor = torch.rand(N,C,H,W)
output_tensor = dropout_layer(input_tensor)

```


##### Pytorch convolutions layer 
卷積層專門用於圖像處理，設計用於使用滑動窗口科技對輸入張量應用卷積運算。 在圖像處理的情况下，其中中間數據表示為大小為N、C、H、W的4D張量。torch.nn.Conv2d是
標準選擇：
```python
conv_layer  = torch.nn.Conv2d(
    in_channels,
    out_channels,
    kerenl_size ,
    stride =1,
    padding=0,
    dialation = 1
)
input_tensor = torch.rand(N,C_in,H,W)
output_tensor = conv_layer(input_tensor) # (N, C_out, H_out,W_out)

```

##### Pytorch recurrent layers
```python
rnn = torch.nn.RNN(input_size,hidden_size,num_layers=1,nonlinearity='tanh',
                bias=True,
                batch_first=False,
                dropout = 0,
                 bidirectional=False,
                )
rnn = nn.RNN(H_in, H_out, num_layers)
input_tensor = torch.randn(L, N, H_in)
h0 = torch.randn(D * num_layers, N, H_out)
output_tensor, hn = rnn(input_tensor, h0)
```
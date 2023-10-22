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


## Pytorch model training 
> torch_model.ipynb

伪代码
### Loss function
首先，我们将了解 PL 中可用的不同损失函数。 本节中的损失函数可以从 torch.nn 模块中找到。

#### MES/L2 loss function
```python
loss = nn.MSELoss(reduction='mean')
input = torch.randn(3,5,requires_grad=True)
target = torch.randn(3,5)
output = loss(input,target)

```

#### MAE/L1 loss function
```python
loss = nnL1Loss(reduction='mean')
input = torch.randn(3,5,requires_grad=True)
target = torch.randn(3,5)
output = loss(input,target)

```

#### CE loss function
```python
loss = nn.CrossEntropyLoss(reduction='mean')
input = torch.randn(3,5,requires_grad=True)
target = torch.randn(3,dtype=torch.long).random_(5)
output = loss(input,target)

```

#### BCE loss function
```python
loss = nn.BCEWithLogitsLoss(reduction='mean')
input = torch.randn(3,requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(input,target)
```

#### custome loss function
```python
def custome_mse_loss(output,target):
    loss = torch.mean((output - target  )**2)
    return loss 
input = torch.randn(3,5,requires_grad=True)
target = torch.randn(3,5)
output = custome_mse_loss(input,target)

```

### Pytorch optimizers 

#### Pytorch SGD optimizer
```python
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,nesterov=True)
```

#### Pytorch Adam optimizer
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 
```

## Implementing and training a model in TF 
PyTorch 面向研究项目，而 TF 更注重行业使用
案例。 虽然 PyTorch、Torch Serve 和 Torch Mobile 的部署功能仍处于实验阶段，但 TF、TF Serve 和 TF Lite 的部署功能稳定且正在积极使用。 TF 的第一个版本由 Google Brain 团队于 2011 年推出，他们一直在不断更新 TF，使其更加灵活、用户友好和高效。

TF 和 PyTorch 之间的主要区别最初要大得多，因为 TF 的第一个版本使用静态图。 然而，这种情况在版本 2 中发生了变化，因为它引入了急切执行，模仿 PyTorch 中已知的动态图。 TF 版本 2 通常与 Keras 一起使用，Keras 是 ANN 的接口 (https://keras.io)。 Keras 允许用户快速开发深度学习模型并运行实验。 在以下部分中，我们将引导您了解 TF 的关键组件。

### TF data loading logic 
```python
import tensorflow_datasets as tfds 
class DataLoader:
    @staticmethod
    def load_data(config):
        return tfds.load(config.data_url)

```
另外一种方法
```python
import tensorflow as tf 
dataset = tf.data.TFRecordDataset(list_of_files)
dataset = tf.data.Dataset.from_tensor_slices((df_features.values,df_target.values))
def data_generator(images,labels):
    def fetch_example():
        i = 0 
        while True:
            example = (images[i],labels[i])
            i +=1 
            i%= len(labels)
            yield example 
        return fetch_examples 
training_dataset = tf.data.Dataset.from_generator(
    data_generator(images,labels),
    output_types = (tf.float32,tf.int32),
    output_shapes = (tf.TensorShape(features_shape),tf.TensorShape(labels_shape))
)

```
### TF model definition
与 PyTorch 和 PL 处理模型定义的方式类似，TF 提供了多种定义网络架构的方法。 首先，我们将了解 Keras.Sequential，它链接一组层来构建网络。 此类为您处理链接，以便您无需显式定义层之间的链接：
```python
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
input_shape = 50 
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(128,activation='relu',name='layer1'),
        layers.Dense(64,activation='relu',name='layer2'),
        layers.Dense(1,activation='sigmoid',name='layer3'),
    ]
)
```
在前面的示例中，我们创建的模型由一个输入层、两个密集层和一个生成单个神经元作为输出的输出层组成。 这是一个可用于二元分类的简单模型。

如果模型定义比较复杂，无法按顺序构建，另一种选择是使用 keras.Model 类，如以下代码片段所示：
```python
num_classes = 5 
input_1 = layers.Input(50)
input_2 = layers.Input(10)
x_1 = layers.Dense(128,activation='relu',name='layer1x')(input_1)
x_1 = layers.Dense(64,activation='relu',name='layer1_2x')(x_1)
x_2 = layers.Dense(128,activation='relu',name='layers2x')(input_2)
x_2 = layers.Dense(64,activation='relu',name='layer2_1x')(x_2)
x = layers.concatenate([x_1,x_2],name='concatenate')
out = layers.Dense(num_classess,activation='softmax',name='output')(x)
model = keras.Model((input_1,input_2),out)
```
在此示例中，我们有两个输入以及一组不同的计算。 两条路径在最后一个串联层中合并，将串联张量传输到具有五个神经元的最终密集层。 鉴于最后一层使用softmax激活，该模型可以用于多类分类。


第三个选项如下，是创建一个继承keras.Model的类。 此选项为您提供最大的灵活性，因为它允许您自定义模型的每个部分和训练过程：

```python
class SimpleANN(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_1 = layers.Dense(128,activation='relu',name='layer1')
        self.dense_2 = layers.Dense(64,activation='relu',name='layer2')
        self.out = layers.Dense(1, activation="sigmoid",name="output")
    def call(self,inputs):
        x  = self.dense_1(inputs)
        x = self.dense_3(x)
        return self.out(x)
    def build_graph(self,raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return keras.Model(inputs=[x],outputs=self.call(x))
model = SimpleANN()
```

### TF DL layers 
如上一节所述，tf.keras.layers 模块提供了一组可用于构建 TF 模型的层实现。 在本节中，我们将介绍在 PyTorch 中实现和训练模型部分中描述的同一组层。

####  TF dense (linear) layers

```python
tf.keras.layers.Dense(units, activation=None, use_bias=True,
kernel_initializer='glorot_uniform', bias_initializer='zeros',
kernel_regularizer=None, bias_regularizer=None, activity_
regularizer=None, kernel_constraint=None, bias_constraint=None,**kwargs)
x = layers.Dense(128,name='layer2')(input)
x = tf.keras.layers.Activation('relu')(x)

```

在某些情况下，您需要构建自定义层。 以下示例演示了如何通过继承tensorflow.keras.layers，使用基本的TF操作创建密集层。 层类：
```python
import tensorflow as tf 
from tensorflow.keras.layers import Layer 
class CustomDenseLayer(Layer):
    def __init__(self,units=32):
        super(SimpleDense,self).__init__()
        self.units = units 
    def build(self,input_shape):
        w_init = tf.random_normal_initiliazr()
        self.w = tf.Variable(name="kernel",
                            initial_value=w_init(shape=(input_shape[-1], self.units),
                            dtype='float32'),trainable=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',initial_value = b_init(shape=(self.units),dtype='float32',),trainable =True)
    def call(self,inputs):
        return tf.matumul(inputs,self.w)+self.b 
```

####  TF pooling layers
```python
tf.keras.layers.MaxPool2D(
                        pool_size=(2, 2), strides=None, padding='valid', data_
                        format=None,
                        kwargs)
tf.keras.layers.AveragePooling2D(
                        pool_size=(2, 2), strides=None, padding='valid', data_
                        format=None,
                        kwargs)
                        
```

####  TF normalization layers 
```python
tf.keras.layers.BatchNormalization(
                        axis=-1, momentum=0.99, epsilon=0.001, center=True,
                        scale=True,
                        beta_initializer='zeros', gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones', beta_regularizer=None,
                        gamma_regularizer=None, beta_constraint=None, gamma_
                        constraint=None, **kwargs)
```

#### TF dropout layers 
#### TF convulition layers 
#### TF recurrent  layers 


## TF model trianing 
> tensorflow_model.ipynb 
>


## Decomposing a complex,state-of-the-art model implementation
### StyleGAN
StyleGAN 是生成对抗网络 (GAN) 的一种变体，旨在从潜在代码（随机噪声向量）生成新图像。

其架构可以分为三个元素：映射网络、生成器和鉴别器。 在较高层次上，映射网络和生成器协同工作，从一组随机值生成图像。 鉴别器在训练过程中指导生成器生成真实图像方面发挥着关键作用。 让我们仔细看看每个组件。

### Implementation in Pytorch 
不幸的是，NVIDIA 尚未分享 PyTorch 中 StyleGAN 的公开实现。 相反，他们发布了 StyleGAN2，它共享大部分相同的组件。 因此，我们将在 PyTorch 示例中使用 StyleGAN2 实现：https://github.com/NVlabs/stylegan2-ada-pytorch。


所有网络组件都可以在training/network.py 下找到。 这三个组件的命名如上一节所述：MappingNetwork、Generator 和 Discriminator。

#### Model training logic for Pytorch
```python
def training_loop(...):
    .....
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set,sampler=training_set_sampler,batch_size = batch_size//num_gpus,**cuda_loader_kwargs))
    loss = dunnlib.util.construct_class_by_name(device=device,**ddp_modules,**loss_kwargs)
    while True:
        # fetch training data
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img,phase_real_c = next(training_set_iterator)
        # execute training phases 
        for phase,phase_gen_z,phase_gen_c in zip(phase,all_gen_z,all_gen_c):
            # accumulate gradients over multiple rounds 
            for round_idx,(real_img,real_c,gen_z,gen_c) in enumerate(zip(phase_real_img,phase_real_c,phase_gen_z,phase_gen_c)):
                loss.accumulate_gradients(phase=phase.name, real_
                        img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                        sync=sync, gain=gain)
        # update weights 
        phase.module.requires_grad_(False)
        with torch.autograd.profiler.record_function(phase.name+'_opt'):
            phase.opt.step()


```

### Implementation in tensorflow 

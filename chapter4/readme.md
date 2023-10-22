# Experiment Tracking Model Management and Dataset Versioning 
## Overview of DL project tracking 
訓練深度學習模型是一個迭代過程，會消耗大量時間和資源。 因此，保持
追蹤所有實驗並持續組織它們可以防止我們浪費時間和不必要的操作。

### Components of DL project tracking 
最主要的就是**experiment tracking** ,**model management**,**dataset versioning**

#### Experiment tracking 
實驗追蹤背後的概念很簡單：儲存每個實驗的描述和動機，這樣我們就不會出於相同目的運行另一組實驗。

#### Model management 
模型管理超越了實驗跟踪，因為它涵蓋了模型的整個生命週期：資料集
資訊、工件（訓練模型產生的任何資料）、模型的實現、評估指標和管道資訊（例如開發、測試、登台和生產）。



模型管理使我們能夠快速選取感興趣的模型並有效率地建立模型
模型可以使用的環境。


#### Dataset versioning 
深度學習專案追蹤的最後一個組成部分是資料集版本控制。 在許多專案中，資料集會發生變化
時間。 變更可能來自資料模式（資料組織方式的藍圖）、檔案位置、
甚至來自應用於資料集的過濾器來操縱基礎資料的含義。 許多
行業中發現的數據集以複雜的方式構建，並且通常存儲在多個位置
以各種資料格式。 因此，變化可能比你想像的更劇烈、更難追蹤。預計。

#### Tools for DL project tracking 
- TensorBoard
- Weights/Biases
- Neptune 
- MLflow 
- SageMaker
- Kubeflow 
- Valohai


### DL project tracking with weights & biases
W&B 是一個實驗管理平台，提供模型和數據的版本控制。

#### wandb simple tutorial
```
# 命令行
pip install wandb 
wandb login
##python 代碼
import wandb 
run_1 = wandb.init(project='xxx',name='rrr') 

```
建立運行後，您可以開始記錄資訊； wandb.log 函數允許您可以記錄任何您想要的數據。

wandb.config 是追蹤模型超參數的絕佳位置。 對於任何來自
實驗中，可以使用wandb.log_artifact方法（更多詳情請見https://docs.wandb.ai/guides/artifacts）。 記錄工件時，您需要定義檔案路徑，然後指派工件的名稱和類型，如下列程式碼片段所示：

```python
wandb.log_artifact(file_path,name='new_artifact',type='my_dataset')
run = wandb.init(project = 'example-DL-Book')
artifact = run.use_artifact('example-DL-Book/new_artifact:v0',type='my_dataset')
artifact_dir = artifact.download()
```

**Integrating W&B into a keras project**
```python
import wandb 
from wandb.keras import WandbCallback
from tensorflow import keras 
from tensorflow.keras import layers
wandb.init(project="example-DL-Book", name="run-1")
wandb.config = {
   "learning_rate": 0.001,
   "epochs": 50,
   "batch_size": 128
}
model = keras.Sequential()
logging_callback = WandbCallback(log_evaluation=True)
model.fit(
    x = x_train,
    y = y_train,
    epochs = wandb.config['epochs'],
    batch_size = wandb.config['batch_size'],
    verbose = 'auto',
    validation_data = (x_valid,y_valid),
    callbacks = [logging_callback],
)

```

**Integrating W&B into a Pytorch Lightning project**
```python
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="example-DL-Book")
trainer = Trainer(logger=wandb_logger)
class LitModel(LightningModule):
    def __init__(self,*args,**kwargs):
        self.save_hyperparameters()
    def training_step(self, batch, batch_idx):
       self.log("train/loss", loss)
```

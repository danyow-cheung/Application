# Deep Learning on Mobile Devices 


## Preparing DL models for mobile devices 
轉換成onnx

### Generating a torchscript model 

```python
import torch 
from torch.utils.mobile_optimizer import optimize_for_mobile

# load a trained pytorch model 
saved_model_file = 'model.pt'
model = torch.load(saved_model_file)

model.eval()
# convert 
torchscript_model = torch.jit.script(model)
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
# save 
torch.jit.save(torchscript_model_optimized,'mobile_optimized.pt')
```

## Creating IOS apps with a DL model 
### Running TorchScript model inference on iOS
我们将从Swift代码片段开始，该代码片段使用TorchModule模块来加载经过训练的TorchScript模型。

添加下面的信息到`podfile`
```
pod 'LibTorch_Lite','~>1.10.0'
```
As described in the last section, you can run the pod install command to install the library.

假设TorchScript模型是为C++设计的，Swift代码无法运行模型推理
直接地为了弥补这一差距，TorchModule类是
torch:：jit:：mobile:：模块。若要在应用程序中使用此功能，请创建名为
TorchBridge需要在项目下创建，并且包含TorchModule.mm
（Objective-C实现文件）、TorchModule.h（头文件）和桥接头文件
使用-Bridging Header.h后缀的命名约定（以允许Swift加载
Objective-C库）。完整的示例设置可在
https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld/HelloWorld/HelloWorld/TorchBridge.
```c++
// import the TorchModule class to the project:
#include "TorchModule.h"
// instantiate torchmodule by providing a path to the torchscript model file 
let modelPath = "model_dir/torchscript_model.pt"
let module = TorchModule(modelPath: modelPath)

// inference 
let inputData:Data 
inputData = ...
let outputs = module.predict(input:UnsafeMutableRawPointer(&inputData))
```

## Creating Android apps with a DL model

### Running TorchScript model inference on Android

在本节中，我们将解释如何在Android应用程序中运行TorchScript模型。要在Android应用程序中运行TorchScript模型推理，您需要由org.pytorch:pytorch_Android_lite库提供的Java包装器。同样，您可以指定必要的库
在.gradle文件中，如以下代码片段所示：

```
dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.11'
}
```

Running TorchScript model inference in an Android app can be achieved by following the steps
presented next. The key is to use the Module class from the org.pytorch library, which calls a
C++ function for inference behind the scenes (https://pytorch.org/javadoc/1.9.0/
org/pytorch/Module.html)

```c++
import org.pytorch.Module;
let torchscript_model_path = "model_dir/torchscript_model.pt";
Module = Module.load(torchscript_model_path);
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();


```
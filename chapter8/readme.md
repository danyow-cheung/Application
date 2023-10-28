# Simplifying Deep Learning Model Deployment

## Introduction to ONNX 

### running inference using ONNX Runtime 
```python
import onnxruntime as rt
providers = ['CPUExecutionProvider'] # select desired provider or use rt.get_available_providers()

model = rt.InferenceSession("model.onnx", providers=providers)

onnx_pred = model.run(output_names, {"input": x}) # x is your model's input

```

## Conversion between Pytorch and ONNX 
### Converting a PyTorch model into an ONNX model
Interestingly, PyTorch has built-in support for exporting its model as an ONNX model
```python
import torch
torch_model = ''
dummy_input = torch.randn(...,requires_grad=True)
onnx_model_path = 'model.onnx'
# export to the model
torch.onnx.export(
    torch_model,dummy_input,onnx_model_path
)
```

### Converting an ONNX model into a PyTorch model
```python
import onnx 
from onnx2pytorch import ConvertModel

onnx_model = onnx.load("model.onnx")
pytorch_model = ConvertModel(onnx_model)

```
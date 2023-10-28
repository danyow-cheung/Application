# Revealing the secret of Deep Learning Models
我们将介绍超参数调整，这是找到正确训练配置的最标准过程

我们还将研究可解释人工智能领域，这是了解模型在预测过程中的作用。

我们将介绍三种最常见的技术
在这个领域：置换特征重要性（PFI），SHapley加性解释（SHAP），局部可解释模型不可知解释（LIME）。

## Obtaining the best performing model using hyperparameter tuning
使用超参数调整获得性能最佳的模型

~~在机器学习（ML）中，超参数是指控制学习过程的任何参数。 在许多情况下，数据科学家通常关注与模型相关的超参数，如特定类型的层、学习率或优化器类型。然而，超参数还包括,与数据相关的配置，例如要应用的增强类型和模型的采样策略,训练改变一组超参数的迭过程，以及理解性能,为目标任务找到正确的超参数集称为超参数调整。确切地说，您将有一组想要探索的超参数。对于每次迭代，
一个或多个超参数将以不同的方式配置，并且将使用调整后的设置。在迭代过程之后，用于最佳模型将是最终的输出。~~


### Hyperparameter tuning techniques 
- Grid search
- Random search
- Bayesian optimization

## Understanding the behavior of the model with Explainable Ai
可解释人工智能是一个非常活跃的研究领域。在商业环境中，了解人工智能模型可以很容易地获得独特的竞争优势
### Permutation Feature Importance
神经网络缺乏理解输入特征对预测（模型输出）的影响所需的内在属性。然而，有一种模型不可知的方法，称为置换特征重要性（PFI），是为解决这一困难而设计的


进一步说，我们可以完全去除这个特性，并测量模型性能。
这种方法被称为特征重要性（FI），也称为排**列重要性（PI**）或**平均减少准确性（MDA）**。让我们看看如何为任何黑匣子型号实现FI

#### Feature Importance
使用到`ELI5` 表示FI分析
```python
import eli5
from eli5.sklearn import PermutationImportance
def score(self,x,y_true):
    y_pred = model.predict(x)
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred-y_true),axis=-1))
perm = PermutationImportance(model,tandom_state=1,scoring=score).fit(features,labels)
fi_perm = perm.feature_importances_
fi_std = perm.feature_importances_std_
```
如果该模型既不基于scikit learn也不基于Keras，则需要使用permutation_importance.get_score_importance函数。以下代码片段描述了如何在PyTorch模型中使用此函数：
```python
import numpy as np 
from eli5.permutation_importance import get_score_importances 
black_box_model = "" 
def score(x,y):
    y_pred = black_box_model.predict(x)
    return accuracy_score(y,y_pred)
base_score, score_decreases = get_score_importances(score, x,y)
feature_importances = np.mean(score_decreases, axis=0)

```

### SHapley Additive exPlanations(SHAP)
SHAP是一种利用Shapley值来理解给定黑盒的解释方法模型
```python
import shap 
shap.initjs()
model = ''
explainer = shap.KernelExplainer(model,sampled_data)
shap_values = explainer.shap_values(data, nsamples=300)
shap.force_plot(explainer.expected_value, shap_values, data)
shap.summary_plot(shap_values, sampled_data, feature_names=names, plot_type="bar")


```
### Local Interpretable Model-agnostic Explanations (LIME)
LIME是一种训练局部代理模型来解释模型预测的方法。
```python
from lime.lime_tabular import LimeTabularExplainer as Lime
from matplotlib import pyplot as plt
expl = Lime(features, mode='classification', class_names=[0,1])
# explain first sample
exp = expl.explain_instance(x[0], model.predict, num_features=5, top_labels=1)
# show plot
exp.show_in_notebook(show_table=True, show_all=False)

```
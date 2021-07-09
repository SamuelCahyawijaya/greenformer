# PyAutoFact
### Automatic factorization library for pytorch

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

If you use any source codes included in this toolkit in your work, please cite the following paper.
- Winata, G. I., Cahyawijaya, S., Lin, Z., Liu, Z., & Fung, P. (2020, May). Lightweight and efficient end-to-end speech recognition using low-rank transformer. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6144-6148). IEEE.

### What is PyAutoFact
PyAutoFact is a library to convert `Linear`, `Conv1d`, `Conv2d`, `Conv3d` layers into its own variant which called `LED`.
PyAutoFact seeks over your PyTorch module, replace all `Linear` layers into `LED` layers and all `Conv1d`, `Conv2d`, `Conv3d` layers into `CED` layers with the specified rank.

### How to Install
```
pip install PyAutoFact
```

### Usage
##### BERT Model
```
from transformers import BertModel, BertConfig
from py_auto_fact import auto_fact

config = BertConfig.from_pretrained('bert-base-uncased', pretrained=False)
model = BertModel(config=config)

model = auto_fact(model, rank=100, deepcopy=False, ignore_lower_equal_dim=True, fact_led_unit=False)
```

##### VGG Model
```
import torch
from torchvision import models
from py_auto_fact import auto_fact

model = models.vgg16()
model = auto_fact(model, rank=64, deepcopy=False, ignore_lower_equal_dim=True, fact_led_unit=False)
```

### Why Use PyAutoFact
- Improve the speed of you model significantly, check our [Example Notebook](https://github.com/SamuelCahyawijaya/py_auto_fact/blob/main/examples/factorize_bert.ipynb)
- Mantain model performance with appropriate choice of rank, check our [ICASSP 2020 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053878)
- Easy to use and works on any kind of model!

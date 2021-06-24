# PyAutoFact
### Automatic linear factorization library for pytorch

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

If you use any source codes included in this toolkit in your work, please cite the following paper.
- Winata, G. I., Cahyawijaya, S., Lin, Z., Liu, Z., & Fung, P. (2019). Lightweight and Efficient End-to-End Speech Recognition Using Low-Rank Transformer. arXiv preprint arXiv:1910.13923. (Accepted in ICASSP 2020)

### What is PyAutoFact
PyAutoFact is a small library to convert `Linear` layer into its own variant which called `LED`.
PyAutoFact seeks over your PyTorch module and replace all `Linear` layers into `LED` layers with the specified rank

### How to Install
```
pip install PyAutoFact
```

### Usage
```
from transformers import BertModel, BertConfig
from py_auto_fact import auto_fact

config = BertConfig.from_pretrained('bert-base-uncased', pretrained=False)
model = BertModel(config=config)

model = auto_fact(model, rank=100)
```

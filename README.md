# PyAutoFact
Automatic linear factorization library for pytorch

### What is PyAutoFact
Seek over your PyTorch module, and change all `Linear` layers into `LED` layers with specified rank

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

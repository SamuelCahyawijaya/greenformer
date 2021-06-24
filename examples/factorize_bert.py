from transformers import BertModel, BertConfig
from py_auto_fact import auto_fact

# Load Model
config = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel(config=config)

print('== Original Model ==')
print(model)
print()

# Perform Auto Factorization
model = auto_fact(model, rank=100)

print('== Factorized Model ==')
print(model)
print()

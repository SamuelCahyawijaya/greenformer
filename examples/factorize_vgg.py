import torch
from torchvision import models
from py_auto_fact import auto_fact

# Load Model
model = models.vgg16()

print('== Original Model ==')
print(model)
print()

# Perform Auto Factorization
model = auto_fact(model, rank=64)

print('== Factorized Model ==')
print(model)
print()

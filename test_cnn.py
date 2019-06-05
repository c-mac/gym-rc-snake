import torch
import torch.nn as nn

model = nn.Conv2d(3, 20, 3, padding=1)
input = torch.zeros((2, 3, 1, 80))

# print(input)

model(input)

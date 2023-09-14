import torch
from activation import Sqrt

act = Sqrt(slope=2)

a = torch.arange(1.0, 13.0).view(4, 3)
b = act.forward(a)
c = abs(a)

print(a)
print(b)
print(c)

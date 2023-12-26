import torch
from torch import Tensor
import torch.nn as nn

class Sqrt(nn.Module):
	def __init__(self, slope: float = 1.):
		super().__init__()
		self.slope = slope
	
	def sqrt(self, input_tensor: Tensor) -> Tensor:
		self.result = torch.mul(torch.mul(torch.sub(torch.sqrt(torch.add(torch.abs(input_tensor), 0.25)), 0.5), torch.sgn(input_tensor)), self.slope)
		return self.result

	def forward(self, input_tensor: Tensor) -> Tensor:
		return self.sqrt(input_tensor)

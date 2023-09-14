import torch
from torch import Tensor
import torch.nn as nn


class Sqrt(nn.Module):
	def __init__(self, slope: float = 1.):
		super().__init__()
		self.slope = slope

	def forward(self, input_tensor: Tensor) -> Tensor:
		slope_tensor = abs(input_tensor) / input_tensor
		return ((abs(input_tensor) + 0.25) ** 0.5 - 0.5) * self.slope * slope_tensor

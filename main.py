import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from activation import Sqrt


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(20, 15)
		self.fc2 = nn.Linear(15, 10)
		self.activation = nn.LeakyReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.conv3(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.conv4(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(20, 15)
		self.fc2 = nn.Linear(15, 10)
		self.activation = Sqrt(slope=2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.conv3(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.conv4(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x


def train(model, train_data, error_func, lr):
	optimizer = torch.optim.SGD()



torch.manual_seed(2052)
train_data = datasets.MNIST(
	"./data/",
	train=True,
	download=True,
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
)
train_loader = train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(
	'./data/',
	train=False,
	transform=transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize((0.1307,), (0.3081,))
	])
)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

model = CNN()

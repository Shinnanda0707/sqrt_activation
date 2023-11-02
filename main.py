from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader, random_split
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


class SqrtCNN(nn.Module):
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


def train(model, train_data, loss_func, optimizer):
	model.train()

	train_loss = 0
	train_acc = 0
	total = 0

	for i, data in enumerate(train_data):
		optimizer.zero_grad()
		x, y_hat = data
		x = x.cuda()
		y_hat = y_hat.cuda()
		y = model(x)

		loss = loss_func(y, y_hat)
		loss.backward()
		optimizer.step()

		_, prediction = torch.max(y.data, 1)
		train_loss += loss.item()
		total += y_hat.size(0)
		train_acc += (prediction == y_hat).sum().item()

	train_loss /= len(train_loader)
	train_acc /= total

	return model, train_loss, train_acc


def val(model, val_data, loss_func, optimizer):
	net.eval()

	val_loss = 0
	val_acc = 0
	total = 0

	with torch.no_grad():
		for data in val_data:
			x, y_hat = data
			x = x.cuda()
			y_hat = y_hat.cuda()
			y = model(x)

			loss = loss_func(y, y_hat)
			_, prediction = torch.max(y.data, 1)
			val_loss += loss.item()
			val_acc += (prediction == y_hat).sum().item()


parser = ArgumentParser()
parser.add_argument("-bs", "--batch_size", dest="bs", default=128)
parser.add_argument("-nw", "--num_workers", dest="num_workers", default=3)
parser.add_argument("--ms", "--manual_seed", dest="seed", default=123)
parser.add_argument("-clr", "--cnn_learning_rate", dest="clr", default=1e-3)
parser.add_argument("-slr", "--sqrt_cnn_learning_rate", dest="slr", default=1e-3)
args = parser.parse_args()

torch.manual_seed(2052)
transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(
	"./data/",
	train=True,
	download=True,
	transform=transforms
)
test_dataset = datasets.MNIST(
	"./data/",
	train=False,
	download=True,
	transform=transforms
)
train_dataset, val_dataset = random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(args.seed))

train_loader = Dataloader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
val_loader = Dataloader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
test_loader = Dataloader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

print(len(train_dataset))
print(len(test_dataset))

cnn = CNN()
cnn_optim = optim.SGD(cnn.parameters(), lr=args.clr)
sqrt = SqrtCNN()
sqrt_optim = optim.SGD(sqrt.parameters(), lr=args.slr)
loss = nn.CrossEntropyLoss()

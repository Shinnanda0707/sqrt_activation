import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from activation import Sqrt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 100)
        self.fc4 = nn.Linear(100, 10)
        self.activation = nn.Tanh()

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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x


class SqrtCNN(nn.Module):
    def __init__(self, slope):
        super(SqrtCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 100)
        self.fc4 = nn.Linear(100, 10)
        self.activation = Sqrt(slope=slope)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation.forward(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.activation.forward(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.activation.forward(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.activation.forward(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation.forward(x)
        x = self.fc2(x)
        x = self.activation.forward(x)
        x = self.fc3(x)
        x = self.activation.forward(x)
        x = self.fc4(x)
        return x


dataset = datasets.MNIST(
	"./data/",
	train=True,
	download=True,
	transform=transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.1307,), (0.3081,))
    ])
)[0]
data, _ = DataLoader(dataset)

cnn = CNN()
sqrt = SqrtCNN(1)

tb = SummaryWriter("runs/Model_graphs/Tanh")
tb.add_graph(cnn, data)
tb.close()

tb = SummaryWriter("runs/Model_graphs/Sqrt")
tb.add_graph(sqrt, data)
tb.close()

import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from train_val import train, val

torch.cuda.empty_cache()


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


parser = ArgumentParser()
parser.add_argument("-bs", "--batch_size", dest="bs", type=int, default=128)
parser.add_argument("-nw", "--num_workers", dest="num_workers", type=int, default=3)
parser.add_argument("-ms", "--manual_seed", dest="seed", type=int, default=123)
parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=100)
parser.add_argument("-lr", "--learning_rate", dest="learning_rate", type=float, default=1e-4)
args = parser.parse_args()

torch.manual_seed(args.seed)
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

train_loader = DataLoader(
	train_dataset,
	batch_size=args.bs,
	shuffle=True,
	num_workers=args.num_workers
)
val_loader = DataLoader(
	val_dataset,
	batch_size=args.bs,
	shuffle=False,
	num_workers=args.num_workers
)
test_loader = DataLoader(
	test_dataset,
	batch_size=args.bs,
	shuffle=False,
	num_workers=args.num_workers
)

model = CNN()
model.cuda()
# model.load_state_dict(torch.load("/home/pie/Desktop/Python/AI/SQRT_activation/CNN_init.pth"))
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
loss = nn.CrossEntropyLoss()

tb = SummaryWriter(os.path.join("/home/pie/Desktop/Python/AI/SQRT_activation/runs", f"Tanh_{args.learning_rate}"))
print(f"{'=' * 40}[Start training CNN with lr={args.learning_rate}]{'=' * 40}")
st = time.time()

for i in range(args.epoch):
    print(i + 1)
    model, train_loss, train_acc = train(model, train_loader, loss, optimizer)
    val_loss, val_acc = val(model, train_loader, loss)
    
    torch.save(model.state_dict(), os.path.join(f"/home/pie/Desktop/Python/AI/SQRT_activation/Tanh/{args.learning_rate}", f"{i + 1}.pth"))
    tb.add_scalar("Train Loss", train_loss, i + 1)
    tb.add_scalar("Train Accuracy", train_acc, i + 1)
    tb.add_scalar("Validation Loss", val_loss, i + 1)
    tb.add_scalar("Validation Accuracy", val_acc, i + 1)
tb.close()

et = round(time.time() - st)
print(f"Done! Took {et // 60}min {et % 60}s.")

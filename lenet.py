import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu') # ?6
args = parser.parse_args()

class Lenet(nn.module):

    def __init__(self):
        super().__init__()
        self.conv1  = nn.conv2d(1, 6, kernel_size = 5)
        self.conv2  = nn.conv2d(6, 16, kernel_size = 5)
        self.pool1  = nn.MaxPool2d((2, 2), stride = 2) 
        self.pool2  = nn.MaxPool2d((2, 2), stride = 2) 
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        self.relu    = nn.ReLU(inplace = True)
        
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = out.view(-1, 1)
        # full connection 怎么写
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

# train
# load train data and test data
# before loading, we need to do some preprocess.
cifarSet = datasets.CIFAR10(root = "../cifar10", train = True, download = True) # when download, root is the folder to store

model = Lenet()

if args.gpu:
    # use gpu
    model = model.cuda() # important

# loss function
criterion = nn.CrossEntropyLoss() # parameter ?
optimzer  = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)


def train():
    # firstly, you need to transfer the data to cuda
    # then you need to backgrad

    # how to acquire the train data?
    

def test():
    



    

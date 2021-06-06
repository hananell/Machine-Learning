import torch
import sys
import numpy as np
import torchvision
from torch import nn,optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import math #for using floor to part the data from fashionMnist
from torchvision import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 32

class ModelAB(nn.Module):
    def __init__(self, image_size):
        super(ModelAB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x1 = x.view(-1,self.image_size)
        x2 = F.relu(self.fc0(x1))
        x3 = F.relu(self.fc1(x2))
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1)

class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self, x):
        dropout = torch.nn.Dropout(p=0.2)
        x1 = x.view(-1,self.image_size)
        x2 = F.relu(self.fc0(x1))
        x2 = dropout(x2)
        x3 = F.relu(self.fc1(x2))
        x3 = dropout(x3)
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1)

class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x1 = x.view(-1, self.image_size)
        batch = nn.BatchNorm1d(784)
        x1 = batch(x1)
        x2 = F.relu(self.fc0(x1))
        batch1 = nn.BatchNorm1d(100)
        x2 = batch1(x2)
        x3 = F.relu(self.fc1(x2))
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1)

class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
    def forward(self, x):
        x1 = x.view(-1,self.image_size)
        x2 = F.relu(self.fc0(x1))
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = F.relu(self.fc3(x4))
        x6 = F.relu(self.fc4(x5))
        x7 = self.fc5(x6)
        return F.log_softmax(x7, dim=1)

class ModelF(nn.Module):
    def __init__(self, image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
    def forward(self, x):
        x1 = x.view(-1,self.image_size)
        x2 = F.sigmoid(self.fc0(x1))
        x3 = F.sigmoid(self.fc1(x2))
        x4 = F.sigmoid(self.fc2(x3))
        x5 = F.sigmoid(self.fc3(x4))
        x6 = F.sigmoid(self.fc4(x5))
        x7 = self.fc5(x6)
        return F.log_softmax(x7, dim=1)

# def train():
#     model.train()
#     trainLoss = 0.0
#     correct = 0.0
#     for data, labels in train_loader:
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, labels)
#         trainLoss += loss.item()
#         loss.backward()
#         optimizer.step()
#         pred = output.max(1, keepdim=True)[1]
#         correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
#     trainLoss /= len(train_loader)
#     accuracy = correct / len(train_loader)
#     return trainLoss, accuracy

def pytorch_accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(1)
    return (y_pred == y_true).float().mean() * 100

def train():
    model.train()
    trainLoss=0.0
    acc_sum=0.0
    example_counter = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        #loss = F.nll_loss(output, labels.type(torch.int64))
        trainLoss += loss.item()
        loss.backward()
        optimizer.step()
        #pred=output.max(1, keepdim=True)[1]
        acc_sum += float(pytorch_accuracy(output, labels)) * len(data)
        example_counter += len(data)
    trainLoss /= len(train_loader)
    accuracy = acc_sum / example_counter
        # print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #            100. * batch_idx / len(train_loader), loss.item()))
    return trainLoss, accuracy
    print("trainloss= ", trainLoss, "accuracy= ", accuracy,"\n")


def validation():
    model.eval()
    valLoss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)
            valLoss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum().item()
    valLoss /= len(validation_loader)
    accuracy = correct / len(validation_loader)
    return valLoss, accuracy


def modelA():
    model = ModelAB(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr) #@origin
    return model, optimizer

def modelB():
    model = ModelAB(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr) #@origin
    return model, optimizer

def modelC():
    model = ModelC(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    return model, optimizer

def modelD():
    model = ModelD(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    return model, optimizer

def modelE():
    model = ModelE(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    return model, optimizer

def modelF():
    model = ModelF(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    return model, optimizer


def makeLoaders():
    transforming = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    train_split_percent = 0.8
    # get full data
    full_train = torchvision.datasets.FashionMNIST('./datasets/', train=True, download=True, transform=transforming)
    # make shuffled list of indices, and split it
    indices = list(range(len(full_train)))
    np.random.shuffle(indices)
    split = math.floor(train_split_percent * len(full_train))
    # split the full data accordingly
    train_indices = indices[:split]
    train_dataset = Subset(full_train, train_indices)
    valid_indices = indices[split:]
    valid_dataset = Subset(full_train, valid_indices)
    # return loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader

if __name__ == '__main__':
    train_loader, validation_loader = makeLoaders()

    model, optimizer = modelA()
    # model, optimizer = modelB()
    # model, optimizer = modelC()
    # model, optimizer = modelD()
    # model, optimizer = modelE()
    # model, optimizer = modelF()

    trainLoss = []; trainAccuracy = []
    validationLoss = []; validationAccuracy = []
    for epoch in range(4):
        tl, ta = train()
        vl,va = validation()
        trainLoss.append(round(tl,4)); trainAccuracy.append(round(ta,4))
        validationLoss.append(round(vl,4)); validationAccuracy.append(round(va,4))

    with open('results', 'w') as f:
        f.write("A train loss:\n")
        for item in trainLoss:
            f.write("%s\n" % item)
        f.write("\nA validation loss:\n")
        for item in validationLoss:
            f.write("%s\n" % item)
        f.write("\nA train accuracy:\n")
        for item in trainAccuracy:
            f.write("%s\n" % item)
        f.write("\nA validation accuracy:\n")
        for item in validationAccuracy:
            f.write("%s\n" % item)


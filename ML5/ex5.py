import torch
import gcommand_dataset as gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
lr = 0.09
epochs = 13
batchSize = 32


def plotMeasurement(title, trainMeasure, valMeasure):
    epochsList = [i for i in range(epochs)]
    plt.figure()
    plt.title(f"model {title}")
    plt.plot(epochsList, trainMeasure, label="Train")
    plt.plot(epochsList, valMeasure, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    plt.legend()
    plt.savefig(f"model {title}.png")


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(4),

        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8192, 600),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Linear(600, 30)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, -1)


def train():
    model.train()
    trainLoss, trainAccuracy = 0., 0.
    loop = tqdm(trainLoader, total=len(trainLoader), leave=False, desc=f"epoch {epoch} train: ")
    for X, Y in loop:
        X, Y = X.to(device), Y.to(device)
        # predict and train
        output = model(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # save measurements. multiply by number of sample in the batch
        trainLoss += loss.item() * len(X)
        y_pred = output.argmax(1)
        trainAccuracy += (y_pred == Y).float().mean().item() * len(X)
        # update tqdm description
        loop.set_postfix_str(f"loss={loss.item():.3f}  accuracy={(y_pred == Y).float().mean().item():.3f}")
    # divide measurements by number of samples overall
    trainLoss /= len(trainData)
    trainAccuracy /= len(trainData)
    return trainLoss, trainAccuracy


def validation():
    model.eval()
    valLoss, valAccuracy = 0., 0.
    with torch.no_grad():
        loop = tqdm(valLoader, total=len(valLoader), leave=False, desc=f"epoch {epoch} validation: ")
        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            valLoss += F.nll_loss(output, Y).item() * len(X)
            y_pred = output.argmax(1)
            valAccuracy += (y_pred == Y).float().mean().item() * len(X)
            loop.set_postfix_str(f"loss={F.nll_loss(output, Y).item() / len(X):.3f}  accuracy={(y_pred == Y).float().mean().item():.3f}")
    valLoss /= len(valData)
    valAccuracy /= len(valData)
    return valLoss, valAccuracy


def test():
    model.eval()
    with open("../../Downloads/test_y", 'w') as f:
        with torch.no_grad():
            for spec, (x, y) in zip(testData.spects, testLoader):
                x = x.to(device)
                output = model(x)
                y_pred = output.argmax(1)
                fileName = spec[0].split('\\')[-1]
                f.write(fileName + ',' + classes[y_pred] + "\n")


if __name__ == '__main__':
    # prepare gcommands
    trainData = gc.GCommandLoader('./gcommands/train')
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    valData = gc.GCommandLoader('./gcommands/valid')
    valLoader = DataLoader(valData, batch_size=batchSize, shuffle=True)
    testData = gc.GCommandLoader('./gcommands/test2')
    # testData.spects = sorted(testData.spects, key=lambda x: int(x[0].split('\\')[-1][:-4]))
    testLoader = DataLoader(testData)
    classes = trainData.classes
    # prepare model
    model = myModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # train and validation
    trainLosses, trainAccuracies, valLosses, valAccuracies = [], [], [], []
    for epoch in range(epochs):
        trainLoss, trainAccuracy = train()
        valLoss, valAccuracy = validation()
        trainLosses.append(trainLoss)
        trainAccuracies.append(trainAccuracy)
        valLosses.append(valLoss)
        valAccuracies.append(valAccuracy)

    # graphs
    plotMeasurement("Loss", trainLosses, valLosses)
    plotMeasurement("Accuracy", trainAccuracies, valAccuracies)

    # test
    test()

    running_time = (float(time.time()) - float(start_time)) / 60
    print(f"---- {running_time:.2f} minutes ---- " )

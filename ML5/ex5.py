# Israel Cohen, Hananel Hadad
# 205812290, 313369183


import os
import torch
import gcommand_dataset as gc
import torch.nn as nn
import torch.nn.functional as F

# hyper parameters
lr = 0.05
trainEpochs = 13
batchSize = 32


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(7680, 400)
        self.fc2 = nn.Linear(400, 30)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.nn.functional.log_softmax(x, -1)


def train(model, dataset):
    model.train()
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
    for sample, label in trainLoader:
        optimizer.zero_grad()
        output = model(sample)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()


def test(model, data, classes):
    model.eval()
    outputList = []
    # predict to outputList
    testLoader = torch.utils.data.DataLoader(data)
    for i, (sample, label) in enumerate(testLoader):
        output = model(sample)
        pred = output.max(1, keepdim=True)[1]
        predClass = classes[pred]
        fileName = os.path.basename(data.spects[i][0])
        outputList.append(f"{fileName},{predClass}\n")
    # sort and write
    sortedOutput = sorted(outputList, key=lambda line: int(line.split('.')[0]))
    with open('test_y', 'w') as file:
        for line in sortedOutput:
            file.write(line)


def validate(model, validSet):
    model.eval()
    correct = 0
    validation_loader = torch.utils.data.DataLoader(validSet, batch_size=batchSize, shuffle=True)
    with torch.no_grad():
        for sample, label in validation_loader:
            output = model(sample)
            prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += prediction.eq(label.view_as(prediction)).cpu().sum().item()
    accuracy = correct / (len(validation_loader)*batchSize)
    return accuracy


# get classes names out of all samples
def getClasses(data):
    classes = []
    # add first class
    firstClass = data[0][0].split('\\')[1]
    classes.append(firstClass)
    # for each file in data, if it's not like the previous, append it
    for file in data:
        curClass = file[0].split('\\')[1]
        if curClass != classes[-1]:
            classes.append(curClass)
    return classes

if __name__ == '__main__':
    # read data, make model
    trainData = gc.GCommandLoader('train')
    testData = gc.GCommandLoader('test')
    classes = getClasses(trainData.spects)

    model1 = myModel()
    optimizer = torch.optim.SGD(model1.parameters(), lr=lr)

    # train and test
    for i in range(13):
        train(model1, trainData)
    test(model1, testData, classes)
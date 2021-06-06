# 313369183 Hananel Hadad
import sys
import random
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid

# given parameters
featuresNumber = 784
firstLayerLength = featuresNumber
labelsNumber = 10
lastLayerLength = labelsNumber
# hyper parameters
innerLayerLength = 70
epochs = 15
eta = 0.01

def initFiles():
    # parameters should be: train_x train_y
    training_examples, training_labels = sys.argv[1], sys.argv[2]
    taken = random.sample(range(1, 55001), 50000)
    taken.sort()

    with open('train_x_short', 'w') as out1:
        with open('train_y_short', 'w') as out2:
            i = 1
            f1 = open(training_examples); f2 = open(training_labels)
            for line1, line2 in zip(f1,f2):
                if i in taken:
                    out1.write(line1); out2.write(line2)
                i += 1
            f1.close(); f2.close()

        with open('validation_x', 'w') as out1:
            with open('validation_y', 'w') as out2:
                f1 = open(training_examples); f2 = open(training_labels)
                i = 1
                for line1,line2 in zip(f1,f2):
                    if i not in taken:
                        out1.write(line1), out2.write(line2)
                    i += 1
                f1.close(); f2.close()

# shuffle two arrays the same way
def shuffleTwo(X,Y):
    # give s indecies of X (and Y)
    s = np.arange(0, len(X), 1)
    # shuffle indecies
    np.random.shuffle(s)
    # make new arrays and give them shuffled values
    newX = np.empty(X.shape);  newY = np.empty(Y.shape)
    for i in range(len(X)):
        newX[i] = X[s[i]]
        newY[i] = Y[s[i]]
    return newX, newY

# turn y to vector in which all values are 0 except at the index of original y that is 1
def oneHotEncoding(Y):
    Y = Y.astype(int)
    encoded = []
    for y in Y:
        vector = [0 for j in range(labelsNumber)]
        vector[y] = 1
        encoded.append(vector)
    return np.array(encoded)

# do min-max normalization, assuming min is 0 and max is 255
def minMaxNormalization(X):
    return np.divide(X, 255)

# train neuron network with given samples and labels
def train(X, Y):
    w1 = np.random.uniform(-0.08, 0.08, [innerLayerLength, firstLayerLength])
    b1 = np.random.uniform(-0.08, 0.08, [innerLayerLength, 1])
    w2 = np.random.uniform(-0.08, 0.08, [lastLayerLength, innerLayerLength])
    b2 = np.random.uniform(-0.08, 0.08, [lastLayerLength, 1])
    weightsDict = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
    for e in range(epochs):
        X,Y = shuffleTwo(X,Y)
        for x, y in zip(X, Y):
            # reshape - add second dimension 1
            x = x.reshape((firstLayerLength, 1))
            y = y.reshape((labelsNumber, 1))
            # do fprop and calculate loss
            z1,h1,z2,h2 = fprop(x,weightsDict)
            loss = -(np.sum(y * np.log(h2)))
            # do bprop and update weights
            gradientsDict = bprop(x,y,z1,h1,z2,h2,weightsDict)
            weightsDict['w1'] -= eta * gradientsDict['w1']
            weightsDict['b1'] -= eta * gradientsDict['b1']
            weightsDict['w2'] -= eta * gradientsDict['w2']
            weightsDict['b2'] -= eta * gradientsDict['b2']
    return weightsDict

# forward propagation
def fprop(x, weightsDict):
        # propagation and loss
        z1 = np.dot(weightsDict['w1'], x) + weightsDict['b1']
        h1 = sigmoid(z1)
        z2 = np.dot(weightsDict['w2'], h1) + weightsDict['b2']
        h2 = softmax(z2)
        return z1,h1,z2,h2

# back propagation
def bprop(x, y, z1, h1, z2, h2, weightsDict):
    # do derivative
    dz2 = (h2 - y)  # dL/dz2 #10x1
    dw2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2 # 10x784
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(weightsDict['w2'].T, dz2) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dw1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'w1': dw1, 'b1': db1, 'w2': dw2, 'b2': db2}

# predict labels of samples, using given neuron network
def predict(X, netDict):
    prediction = []
    w1 = netDict['w1']
    b1 = netDict['b1']
    w2 = netDict['w2']
    b2 = netDict['b2']
    for x in X:
        x = x.reshape(featuresNumber, 1)
        z1 = np.dot(w1, x) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(w2, h1) + b2
        h2 = softmax(z2)
        prediction.append(np.argmax(h2))
    return prediction

def runTest():
    accuracy = 0
    iters = 1
    for i in range(iters):
        # make files
        initFiles()
        train_x = minMaxNormalization(np.loadtxt("train_x_short"))
        train_y = oneHotEncoding(np.loadtxt("train_y_short"))
        validation_x = minMaxNormalization(np.loadtxt("validation_x"))
        validation_y = np.loadtxt("validation_y").astype(int)
        # make prediction
        learnedNetworkDict = train(train_x, train_y)
        myPrediction = predict(validation_x, learnedNetworkDict)
        # rate prediction
        right = 0; wrong = 0
        for prediction, realY in zip(myPrediction, validation_y):
            if prediction == realY:
                right += 1
            else:
                wrong += 1
        accuracy += right / (right + wrong)
    print(f"accuracy:   {accuracy/iters}")


# read files, normalize samples, do one hot encoding on labels
train_x = minMaxNormalization(np.loadtxt(sys.argv[1]))
train_y = oneHotEncoding(np.loadtxt(sys.argv[2]))
test_x = minMaxNormalization(np.loadtxt(sys.argv[3]))
# train neuron network, than use it to predict test
learnedNetworkDict = train(train_x, train_y)
test_y = predict(test_x,learnedNetworkDict)
# write answer
with open('test_y', 'w') as f:
    for i,prediction in enumerate(test_y):
        f.write(f"{prediction}")
        if i != len(test_y)-1:
            f.write("\n")
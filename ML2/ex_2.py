# Hananel Hadad 313369183
import random
import numpy as np
import sys


def initFiles():
    Xfull = open("train_x.txt", "r")
    XnewTrain = open("train_x_mini.txt", "w")
    Xvalidation = open("validation_x.txt", "w")
    Yfull = open("train_y.txt", "r")
    YnewTrain = open("train_y_mini.txt", "w")
    Yvalidation = open("validation_y.txt", "w")

    for x_i, y_i in zip(Xfull, Yfull):
        rand = random.random()
        if rand < 0.9:
            XnewTrain.write(x_i)
            YnewTrain.write(y_i)
        else:
            Xvalidation.write(x_i)
            Yvalidation.write(y_i)

    Xfull.close()
    Yfull.close()
    XnewTrain.close()
    YnewTrain.close()
    Xvalidation.close()
    Yvalidation.close()


# number of features
featuresNum = 12
featuresNumWithBias = 13
# hyper parameters
perceptronIters = 1000
perceptronLearningRate = 0.01
PAIters = 1
k = 5

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

# add given bias to every sample in X
def addBias(X, bias=0.7):
    # make biased array
    XBiased = np.empty((len(X), featuresNumWithBias))
    # take old data
    for i in range(XBiased.shape[0]):
        for j in range(XBiased.shape[1]-1):
            XBiased[i][j] = X[i][j]
    # add bias
    for i in range(XBiased.shape[0]):
            XBiased[i][featuresNumWithBias-1] = bias
    return XBiased

# return correct label based on prediction values of weight of 0 1 2
def labelFromValue(v0, v1, v2):
    maxi = max(v0, v1, v2)
    if maxi == v0: return 0
    elif maxi == v1: return 1
    return 2

# return lists of min and max of each feature
def minMaxLists(X):
    minVals = [9999 for i in range(featuresNum)]
    maxVals = [0 for i in range(featuresNum)]
    for x_i in range(len(X)):
        for feature in range(featuresNum):
            if X[x_i][feature] < minVals[feature]:
                minVals[feature] = X[x_i][feature]
            if X[x_i][feature] > maxVals[feature]:
                maxVals[feature] = X[x_i][feature]
    return minVals, maxVals

class minMaxNormalizer:
    def learn(self,X):
        self.minVals, self.maxVals = minMaxLists(X)

    # normalize all samples in X, according to given min and max lists
    def minMaxNormalizeAll(self, X):
        for x_i in range(len(X)):
            for feature in range(featuresNum):
                curVal = X[x_i][feature]
                minVal = self.minVals[feature]
                maxVal = self.maxVals[feature]
                X[x_i][feature] = self.minMaxNormalizeOne(curVal, minVal, maxVal)

    # normalize single sample, according to its min and max
    def minMaxNormalizeOne(self, x, min, max):
        if max == min: return x
        return (x-min)/(max-min)

class zScoreNormalizer:
    def learn(self, X):
        # featureValues[i] contains all values of feature i from all samples
        featureValues = np.array([np.empty(len(X)) for i in range(len(X[0]))])
        for i in range(len(X)):
            for j in range(len(X[0])):
                featureValues[j][i] = X[i][j]

        # make mean and std from each featureValues[i]
        self.means = np.array([np.mean(featureValues[i]) for i in range(len(featureValues))])
        self.stds = np.array([np.std(featureValues[i]) for i in range(len(featureValues))])

    # normalize all sample, z-score method
    def zScoreNormalizeAll(self, X):
        for i in range(len(X)):
            for j in range(len(X[0])):
                # don't touch wine type, because it's not normal distribution
                if j == 11: continue
                # normalize all others
                X[i][j] = self.zScoreNormalizeOne(X[i][j], self.means[j], self.stds[j])

    # normalize one sample, z-score method
    def zScoreNormalizeOne(self, x, mean, std):
        return (x-mean)/std


class Perceptron:
    def __init__(self):
        # all weights start at 0.01, except for the bias which start at 0
        weightsInit = np.array([1.0 for i in range(featuresNumWithBias)])
        weightsInit[featuresNumWithBias-1] = 0
        self.dictWeights = {0: weightsInit.copy(), 1: weightsInit.copy(), 2: weightsInit.copy()}

    # given samples and right labels, make weights and bias that would be right on them
    def learn(self, X, Y):
        for _ in range(perceptronIters):
            for i,x_i in enumerate(X):
                # for each sample, make prediction and update weight in case of mistake
                prediction = self.predictOne(x_i)
                realY = Y[i]
                # update in case of mistake
                if (prediction != realY):
                    # save bias weight befor updating all weights
                    predictionBiasWeight = self.dictWeights[prediction][featuresNumWithBias-1]
                    realYBiasWeight = self.dictWeights[realY][featuresNumWithBias-1]
                    # update all weights
                    self.dictWeights[prediction] -= perceptronLearningRate * x_i
                    self.dictWeights[Y[i]] += perceptronLearningRate * x_i
                    # correct update bias weight
                    self.dictWeights[prediction][featuresNumWithBias-1] = predictionBiasWeight - perceptronLearningRate
                    self.dictWeights[realY][featuresNumWithBias - 1] = realYBiasWeight + perceptronLearningRate


    def predictAll(self, X):
        # make prediction
        prediction = []
        for x_i in X:
            prediction.append(self.predictOne(x_i))
        return prediction

    # predict labels for given sample, using weights already learned
    def predictOne(self, x):
        prediction0 = np.dot(x, self.dictWeights[0])
        prediction1 = np.dot(x, self.dictWeights[1])
        prediction2 = np.dot(x, self.dictWeights[2])
        prediction = labelFromValue(prediction0, prediction1, prediction2)
        return prediction


class KNN:
    def learn(self, X, Y):
        self.X = X
        self.Y = Y

    def predictAll(self, X):
        return np.array([self.predictOne(x) for x in X])

    def predictOne(self, x):
        # distances from x to all points
        distances = [np.linalg.norm(x-x_i) for x_i in self.X]
        # sort distances, take indices of minimal k
        k_points_indices = np.argsort(distances)[:k]
        # take labels of these minimal k
        k_points_labels = [self.Y[i] for i in k_points_indices]
        # make prediction
        predicted = np.argmax(np.bincount(k_points_labels))
        return predicted


class PA:
    def learn(self, X, Y):
        n_samples, n_features = X.shape
        # dictionary from label to weights
        weightInit = np.array([0.5 for i in range(len(X[0]))])
        self.dictWeights = {0:weightInit.copy(), 1:weightInit.copy(), 2:weightInit.copy()}
        # for each algorithm iteration, for each sample:
        for _ in range(PAIters):
            for i, x_i in enumerate(X):
                # make prediction, update weight and bias accordingly
                prediction = self.predictOne(x_i)
                realY = Y[i]
                loss = max(0, 1 - self.dictWeights[realY].dot(x_i) + self.dictWeights[prediction].dot(x_i))
                if(loss == 0): continue
                # in case of mistake, update weights
                tau = loss/((2*np.linalg.norm(x_i))**2)
                # save bias weight befor updating all weights
                predictionBiasWeight = self.dictWeights[prediction][featuresNumWithBias - 1]
                realYBiasWeight = self.dictWeights[realY][featuresNumWithBias - 1]
                # update all weights
                self.dictWeights[prediction] -= tau*x_i
                self.dictWeights[realY] += tau*x_i
                # correct update bias weight
                self.dictWeights[prediction][featuresNumWithBias - 1] = predictionBiasWeight - tau
                self.dictWeights[realY][featuresNumWithBias - 1] = realYBiasWeight + tau

    # predict label for list of samples
    def predictAll(self, X):
        # make prediction
        prediction = []
        for x_i in X:
            prediction.append(self.predictOne(x_i))
        return prediction

    # predict label for one sample
    def predictOne(self,x):
        prediction0 = np.dot(x, self.dictWeights[0])
        prediction1 = np.dot(x, self.dictWeights[1])
        prediction2 = np.dot(x, self.dictWeights[2])
        prediction = labelFromValue(prediction0, prediction1, prediction2)
        return prediction


'''
# read train values, deal with wine type
dictWines = {b"W":1.0, b"R":0.0}
X = np.genfromtxt(sys.argv[1], delimiter=',', converters={11:lambda wt:dictWines[wt]})
Y = np.genfromtxt(sys.argv[2])
test = np.genfromtxt(sys.argv[3], delimiter=',', converters={11:lambda wt:dictWines[wt]})
# normalize z score
zScoreNormalizeAll(X)
zScoreNormalizeAll(test)
# add bias
XBiased = addBias(X)
testBiasd = addBias(test)
# initialize algorithms
kani = KNN()
kani.learn(X, Y)
perci = Perceptron()
perci.learn(XBiased, Y)
pai = PA()
pai.learn(XBiased, Y)
#run
for i in range(len(test)):
    print(f"knn: {kani.predictOne(test[i])}, perceptron: {perci.predictOne(testBiasd[i])}, pa: {pai.predictOne(testBiasd[i])}")
'''

def run():
    # run PA
    pai = PA()
    pai.learn(XBiased, Y)
    e = pai.predictAll(validationBiased)
    error = 0
    for i in range(len(e)):
        if ans[i] != e[i]: error += 1
    # print("PA accuracy:  ", (len(e) - error) / len(e))
    PAaccuracy = (len(e) - error) / len(e)

    # run perceptron
    perci = Perceptron()
    perci.learn(XBiased, Y)
    p = perci.predictAll(validationBiased)
    error = 0
    for i in range(len(p)):
        if p[i] != ans[i]: error += 1
    # print("perceptron accuracy:  ", (len(p) - error) / len(p))
    perceptronAccuracy =  (len(p) - error) / len(p)

    # run KNN
    knnModel = KNN()
    knnModel.learn(XBiased, Y)
    kp = knnModel.predictAll(validationBiased)
    error = 0
    for i in range(len(kp)):
        if kp[i] != ans[i]: error += 1
    # print("KNN accuracy: ", (len(kp) - error) / len(kp))
    KNNAccuracy = (len(kp) - error) / len(kp)
    return KNNAccuracy, perceptronAccuracy, PAaccuracy


zScoreKNN = 0; zScorePerceptron = 0; zScorePA = 0;
minMaxKNN = 0; minMaxPerceptron = 0; minMaxPA = 0;
iters = 20
for i in range(iters):
    initFiles()
    # read train values, deal with wine type
    dictWines = {b"W":1.0, b"R":0.0}
    X = np.genfromtxt('train_x_mini.txt', delimiter=',', converters={11:lambda wt:dictWines[wt]})
    Y = np.genfromtxt('train_y_mini.txt')
    validation = np.genfromtxt('validation_x.txt', delimiter=',', converters={11:lambda wt:dictWines[wt]})
    ans = np.genfromtxt('validation_y.txt')
    # normalize z score
    normi = zScoreNormalizer()
    normi.learn(X)
    normi.zScoreNormalizeAll(X)
    normi.zScoreNormalizeAll(validation)
    #shuffle
    X, Y = shuffleTwo(X, Y)
    # add bias
    XBiased = addBias(X)
    validationBiased = addBias(validation)
    #run
    knn, perci, pa = run()
    zScoreKNN+=knn; zScorePerceptron+=perci; zScorePA+=pa

    # read train values, deal with wine type
    X = np.genfromtxt('train_x_mini.txt', delimiter=',', converters={11:lambda wt:dictWines[wt]})
    validation = np.genfromtxt('validation_x.txt', delimiter=',', converters={11:lambda wt:dictWines[wt]})
    # normalize min max
    mini = minMaxNormalizer()
    mini.learn(X)
    mini.minMaxNormalizeAll(X)
    mini.minMaxNormalizeAll(validation)
    # add bias
    XBiased = addBias(X)
    validationBiased = addBias(validation)
    # run
    knn, perci, pa = run()
    minMaxKNN+=knn; minMaxPerceptron+=perci; minMaxPA+=pa

print("z-score:")
print(f"KNN: {zScoreKNN/iters}\tperceptron: {zScorePerceptron/iters}\tPA: {zScorePA/iters}")
print("min max:")
print(f"KNN: {minMaxKNN/iters}\tperceptron: {minMaxPerceptron/iters}\tPA: {minMaxPA/iters}")
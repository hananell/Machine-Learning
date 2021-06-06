# author Hananel Hadad 313369183

import numpy as np
import scipy.io.wavfile
from numpy.linalg import norm
import sys

sample, centroids = sys.argv[1], sys.argv[2]
rate, rawpoints = scipy.io.wavfile.read(sample)
points = np.array(rawpoints.copy(), dtype=np.int16)
centroids = np.loadtxt(centroids)
'''
k = 16
centroids = [[] for x in range(k)]
for i in range(k):
    centroids[i] = [100*i,100*i]
centroids = np.array(list(centroids))
'''

#k is number of centroids
# listC[i] is a list that holds all points belong to centroids[i]
k = len(centroids)
listC =  [[] for x in range(k)]
#dicionary from point index to corresponding centroid index
dict={}
#create empty file for output
f = open("output.txt", "w")

for i in range(30):
    #cost calculates the cost of this iteration
    cost = 0
    f.write(f"[iter {i}]:")
    somethingChanged = False
    #for each point, iterate over each centroid indC,
    #find the minimum distance minD and save the index of corresponding optimal centroid minC
    j = 0
    for p in points:
        minD = np.linalg.norm(p - centroids[0])
        indMinC = 0
        indCurC = 0
        for c in centroids:
            curD = np.linalg.norm(p - c)
            if curD < minD:
                minD = curD
                indMinC = indCurC
            indCurC += 1
        #add p to list of c's points
        listC[indMinC].append(p)
        #update the dictionary
        dict[j] = indMinC
        #add to cost
        cost += minD*minD
        j += 1
    #make cost average
    cost = cost / len(points)
    '''
    print(cost)
    '''
    #update each c to mean of corresponding list
    j = 0
    for c in centroids:
        meanC = np.around(np.mean(listC[j], axis=0))
        if not np.array_equal(c,meanC):
            centroids[j] = meanC
            #if at least one centroid is changed, somethingChanged would be true
            somethingChanged = True
        j += 1
    #write reults of iteration
    centroidsFloats = centroids.astype(float)
    j = 0
    for c in centroidsFloats:
        f.write(f"{str(c)}")
        j += 1
        if not j == k:
            f.write(",")
    f.write("\n")
    #if nothing has been changed, break the loop to end the run
    if somethingChanged == False:
        break
    #if there has been a change, clear listC to start over
    else:
        listC = [[] for x in range(k)]

#make new_values. each point is replaced with corressponding value according to the dictionary
new_Values = [[] for x in range(len(points))]
for i in range(len(new_Values)):
    new_Values[i] = centroids[dict.get(i)]

scipy.io.wavfile.write("compressed.wav", rate, np.array(new_Values, dtype=np.int16))

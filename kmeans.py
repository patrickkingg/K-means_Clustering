#Patrick Wang
from csv import reader
from random import randint
from copy import deepcopy
from sys import argv
import numpy as np

def main():

    inputFile = argv[1]
    k = argv[2]
    k=int(k)
    outputFile = argv[3]

    r = reader(open(inputFile))
    df = np.array([row for row in r if row])
    X, y = df[:, :-1], df[:, -1]
    X = X.astype(np.float)

    X = min_max_norm(X)

    clusters, C = k_means(X, k)

    sse = getSSE(X, k, C, clusters)
    print("SSE is %.2f" % sse)

    clusters = list(map(int, clusters))

    with open(outputFile, 'w') as f:
        for i in clusters:
            f.write(str(i) +'\n')
        f.write("SSE is %.2f" % sse)

def min_max_norm(X):
    res = []
    for col in X.T:
        min = col.min()
        max = col.max()
        col = (col - min) / (max - min)
        res.append(col)

    return np.array(res).T

def findDiff(C, oldC):
    res = 0
    for i in range(len(C)):
        res += np.linalg.norm(C[i] - oldC[i], 2)
    return res

def k_means(X, k):

    p =[]
    index = set()

    for i in range(k):
        ind = randint(0, len(X) - 1)
        while ind in index:
            ind = randint(0, len(X) - 1)
        p.append(X[ind])
        index.add(ind)
    C=np.array(p)

    oldC = np.zeros(C.shape)
    diff = findDiff(C, oldC)
    cluster = np.zeros(len(X))

    while diff != 0:
        for i in range(len(X)):
            dis=[]
            for j in range(len(C)):
                dis.append(np.linalg.norm(X[i] - C[j], 2))
            index = np.argmin(dis)

            cluster[i] = index

        oldC = deepcopy(C)

        for i in range(k):
            if list(X[cluster == i]):
                C[i] = np.mean(X[cluster == i], axis=0)

        diff = findDiff(C, oldC)

    return cluster, C

def getSSE(X, k, C, clusters):
    sse = 0
    for i in range(k):
        for j in X[clusters == i]:
            sse += np.linalg.norm(j - C[i], 2) ** 2

    return sse

if __name__ == '__main__': main()
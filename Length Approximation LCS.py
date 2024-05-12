from cmath import inf
import numpy as np
import copy
import math
import timeit
import os

def calculationHProb(m, sigSize):
    P = np.zeros((m,m))
    for q in range(m):
        for k in range(m):
            if k == 0:
                P[k][q] = 1
            else:
                if k > q:
                    P[k][q] = 0
                else:
                    P[k][q] = ((1/sigSize)*P[k-1][q-1]) + (((sigSize - 1)/sigSize) * P[k][q-1])
    return P

def cleanString(s):
    for i in range(10):
        s = s.replace(str(i), '')
    lenS = len(s)
    for i in range(lenS):
        s = s.strip()
    return s

def getStrings(filename, n, lenalphabet, withspace = False):
    if lenalphabet == 100:
        f = open(filename, 'r', encoding='iso-8859-15')
    else:
        f = open(filename, 'r')
    f.readline()
    _strings = []
    for i in range(n):
        _strings.append(cleanString(f.readline()))
        if withspace == True:
            f.readline()
    return _strings

def feasible(locations, alphabet, S, lenstrings):
    for i in range(lenstrings):
        if S[i][locations[i]:].find(alphabet) == -1:
            return False
    return True

def successor(locations, alphabet, S, lenstrings):
    locs = []
    for i in range(lenstrings):
        locs.append(locations[i] + S[i][locations[i]:].find(alphabet) + 1)
    return locs

def calculateScore(C, S, k, lenstrings, P):
    lens = []
    for i in range(lenstrings):
        lens.append(len(S[i]) - C[i])
    h_value = 1.0
    for i in range(lenstrings):
        h_value = h_value * P[k][lens[i]]
    return h_value

def checkDominance(c, best):
    lenc = len(c)
    for i in range(lenc):
        if c[i] <= best[i]:
            return False
    return True

def kBestListFilter(C,h,kbest):
    kbestlist = []
    htemp = copy.deepcopy(h)
    for i in range(kbest):
        t = -1
        ind = 0
        for j in range(len(h)):
            if h[j] > t:
                ind = j
                t = h[j]
        h[ind] = -1
        kbestlist.append(C[ind])
    lenC = len(C)
    Cnewlist = []
    hnewlist = []
    counter = 0
    for i in range(lenC):
        flag = True
        for j in range(kbest):
            if checkDominance(C[i], kbestlist[j]) == True:
                flag = False
                counter += 1
                break
        if flag == True:
            Cnewlist.append(C[i])
            hnewlist.append(htemp[i])
    return Cnewlist, hnewlist, counter

def keepBetaBest(C,h,Beta):
    _rtn = []
    for i in range(Beta):
        t = -1
        ind = 0
        for j in range(len(h)):
            if h[j] > t:
                ind = j
                t = h[j]
        h[ind] = -1
        _rtn.append(C[ind])
    return _rtn

def findKMin(C,S):
    lens = []
    lenstrings = len(S)
    for i in range(lenstrings):
        lens.append(len(S[i]) - C[i])
    _rtn = min(lens)
    return _rtn

def allmins(C,S):
    lens = []
    lenstrings = len(S)
    for i in range(lenstrings):
        lens.append(len(S[i]) - C[i])
    return lens

def findk(children, len_children, S, lenstrings, lenalphabet):
    k_max = 0
    for i in range(len_children):
        q_max = findKMin(children[i], S)
        if q_max > k_max:
            k_max = q_max

    k_min = math.inf
    for i in range(len_children):
        q_min = findKMin(children[i],S)
        if q_min < k_min:
            k_min = q_min
    if k_min == math.inf:
        k_min = 100
    k_max = k_max * (1.8233 - (0.1588 * np.log(lenstrings)))
    k_min = k_min - 31
    
    k_min = int((k_min) / lenalphabet)
    k_max = int((k_max) / lenalphabet)
    if k_max <= 0:
        k_max = 1
    if k_min <= 0:
        k_min = 1
    return k_max, k_min

def main(lens, lena, dataset, inalphabet, dataset_type, beta, flag = False):
    lenalphabet = lena
    lenstrings = lens
    S = getStrings(dataset, lenstrings, lenalphabet, flag)
    alphabet = inalphabet
    alpha = 20
    k_best = 7
    max_length = []
    for i in range(len(S)):
        max_length.append(len(S[i]))
    max_length = max(max_length)
    P = calculationHProb(max_length, lenalphabet)
    B = []
    B.append(np.zeros(lenstrings, dtype= 'int'))
    lenlcs = 0
    start_time = timeit.default_timer()
    while True:
        lenB = len(B)
        children = []
        h = []
        for i in range(lenB):
            for j in range(lenalphabet):
                if feasible(B[i], alphabet[j], S, lenstrings) == True:
                    children.append(successor(B[i], alphabet[j], S, lenstrings))
        len_children = len(children)
        k_max, k_min = findk(children, len_children, S, lenstrings, lenalphabet)
        if dataset_type == 'correlated':
            k = k_min
        else:
            k = k_max
        for i in range(len_children):
            h.append(calculateScore(children[i], S, k, lenstrings, P))
        if len_children == 0:
            break
        else:
            # children, h, co = kBestListFilter(children, h, k_best)
            if len_children > beta:
                B = keepBetaBest(children,h, beta)
            else:
                B = children
        lenlcs += 1
    stoptime = timeit.default_timer()
    return lenlcs, stoptime - start_time


_path = "C:/Users/Alireza/Documents/research/Longest Common Subsequence/Implementations/H-prob Fast"
os.chdir(_path)


alphabet = [4,20]
lens = [10, 15, 20, 25, 40,60,80,100,150,200]
datasets = ["virus", "random", "rat"]
for i in range(len(datasets)):
    ar = []
    for j in range(len(alphabet)):
        if alphabet[j] == 4:
            inalphabet = ['A', 'C', 'G', 'T']
        if alphabet[j] == 20:
            inalphabet = ['A', 'C', 'D','E','F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
        for k in range(len(lens)):
            filename = str(alphabet[j]) + "_" + str(lens[k]) + "_600." + datasets[i]
            lcs = main(lens[k], alphabet[j], filename, inalphabet, 'uncorrelated', 600, False)
            print(filename, lcs)
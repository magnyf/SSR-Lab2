import numpy as np
import matplotlib.pyplot as pl
from math import log, exp
import collections
import tools2
import copy
from sklearn.mixture import *
from test import *

data = np.load('lab2_data.npz')['data']

phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

prondict = {} 
prondict['o'] = ['ow']
prondict['z'] = ['z', 'iy', 'r', 'ow']
prondict['1'] = ['w', 'ah', 'n']
prondict['2'] = ['t', 'uw']
prondict['3'] = ['th', 'r', 'iy']
prondict['4'] = ['f', 'ao', 'r']
prondict['5'] = ['f', 'ay', 'v']
prondict['6'] = ['s', 'ih', 'k', 's']
prondict['7'] = ['s', 'eh', 'v', 'ah', 'n']
prondict['8'] = ['ey', 't']
prondict['9'] = ['n', 'ay', 'n']

modellist = {}
for digit in prondict.keys():
	modellist[digit] = ['sil'] + prondict[digit] + ['sil']

def concatHMMs2(hmmmodels, namelist):
    N = len(namelist)
    word = {}
    word['name'] = 'o'
    word['startprob'] = [0 for i in range(3 * N + 4)]
    word['startprob'][0] = 1.0
    word['means'] = [[0 for j in range(13)] for i in range(3*N)]
    word['covars'] = [[0 for j in range(13)] for i in range(3*N)]

    #compute means and covars
    for i in range(N):
        for j in range(13):
            word['means'][3*i][j] = hmmmodels[namelist[i]]['means'][0][j]
            word['covars'][3*i][j] = hmmmodels[namelist[i]]['covars'][0][j]
                        
        for j in range(13):
            word['means'][3*i+1][j] = hmmmodels[namelist[i]]['means'][1][j]
            word['covars'][3*i][j] = hmmmodels[namelist[i]]['covars'][1][j]

        for j  in range(13):
            word['means'][3*i+2][j] = hmmmodels[namelist[i]]['means'][2][j]
            word['covars'][3*i][j] = hmmmodels[namelist[i]]['covars'][2][j]


    #compute the tranmat matrice, with your code
    transMat = [phoneHMMs[k]['transmat'] for k in namelist]
    #for k in namelist:
    #    tranMat += [phoneHMMs[k]['transmat']]
    n = 0
    m = 0
    for x in transMat:
        n += len(x)-1
        m += len(x[0])-1
    n += 1
    m += 1

    result = [[0 for y in range(n)] for x in range(m)]

    i0 = 0
    j0 = 0
    for x  in transMat:
        for i in range(len(x)):
            for j in range(len(x[0])):
                result[i0+i][j0+j] = x[i][j]
        i0 += len(x) -1
        j0 += len(x[0])-1
    result[-1][-1]  = 1
    word['transmat'] = result
    return word
        
                 
    
wordHMMsO = concatHMMs2(phoneHMMs, ['sil', 'ow', 'sil'])
trans = concatHMMs(phoneHMMs, modellist['o'])

print(trans == wordHMMsO['transmat'])

lmfcc = example['lmfcc']
#print(tools2.log_multivariate_normal_density_diag(trans, wordHMMsO['means'], wordHMMsO['covars']))

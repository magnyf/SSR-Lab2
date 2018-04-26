import numpy as np
import matplotlib.pyplot as pl
from math import log, exp
import collections
import tools2
import copy
from sklearn.mixture import *
import math

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


## list(sorted(phoneHMMs.keys()))
##['ah', 'ao', 'ay', 'eh', 'ey', 'f', 'ih', 'iy', 'k', 'n', 'ow', 'r', 's', 'sil', 'sp', 't', 'th', 'uw', 'v', 'w', 'z']


##phoneHMMs['ah'].keys()
##['name', 'startprob', 'transmat', 'means', 'covars']

modellist = {}
for digit in prondict.keys():
	modellist[digit] = ['sil'] + prondict[digit] + ['sil']

## ----------------
## 3.2
## ---------------
def concatHMMs(hmmmodels, namelist):
	transMatrices = []
	for key in modellist['o']:
		transMatrices += [phoneHMMs[key]['transmat']]

	n = 0
	m = 0
	for x in transMatrices:
		n += len(x)-1
		m += len(x[0])-1
	n += 1
	m += 1

	result = [[0 for y in range(n)] for x in range(m)]

	i0 = 0
	j0 = 0
	for x in transMatrices:
		for i in range(len(x)):
			for j in range(len(x[0])):
				result[i0+i][j0+j] = x[i][j]
		i0 += len(x)-1
		j0 += len(x[0])-1

	result[-1][-1] = 1

	return result


concatMatO = concatHMMs(phoneHMMs, modellist['o'])


#def creatwordHMM(hmmmodels, namelist):
#        wordHMMO =  {}
#        wordHMMO['name'] = namelist
#        wordHMMO['startprob'] = phoneHMMs['sil']['startprob'] + phoneHMMs['ow']['startprob'] + phoneHMMs['sil']['startprob']
#        wordHMMO['transmat']  = concatHMMs(hmmmodels, namelist)
#        wordHMMO['means'] = phoneHMMs['sil']['means'] + phoneHMMs['ow']['means'] + phoneHMMs['sil']['means']
#        wordHMMO['covars'] = phoneHMMs['sil']['covars'] + phoneHMMs['ow']['covars'] + phoneHMMs['sil']['covars']
#        return wordHMMO

# for x in concatMatO:
# 	print(x)

## ----------------
## 3.3
## ---------------
example = np.load('lab2_example.npz')['example'].item()

## ----------------
## 4.1
## ---------------

# TODO
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
            word['covars'][3*i+1][j] = hmmmodels[namelist[i]]['covars'][1][j]

        for j  in range(13):
            word['means'][3*i+2][j] = hmmmodels[namelist[i]]['means'][2][j]
            word['covars'][3*i+2][j] = hmmmodels[namelist[i]]['covars'][2][j]


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

lmfcc = example['lmfcc']

wordHMMsO = concatHMMs2(phoneHMMs, ['sil', 'ow', 'sil'])

obsloglik =tools2.log_multivariate_normal_density_diag(np.array(lmfcc), np.array(wordHMMsO['means']), np.array(wordHMMsO['covars']))

#print(result - example['obsloglik'])


obsloglikExample = example['obsloglik']
#print(len(obsloglik[0]))

# pl.pcolormesh(np.transpose(obsloglik))
# pl.show()


## ----------------
## 4.2
## ---------------

def log_inf(x):
	y = copy.deepcopy(x)

	if not(isinstance(x[0], collections.Iterable)):
	    # not iterable
	    for i in range(len(x)):
	    	if (x[i] > 0):
	    		y[i] = log(x[i])
	    	else:
	    		y[i] = -float('Inf')
	else:
		# iterable
		for i in  range(len(x)):
			for j in  range(len(x[0])):
				if (x[i][j] > 0):
					y[i][j] = log(x[i][j]) 
				else:
					y[i][j] = -float('Inf')
	return y



def forward(log_emlik, log_startprob, log_transmat):
	N = len(log_emlik)
	M = len(log_emlik[0])

	logAlpha = [[0 for x in range(M)] for y in range(N)]

	for j in range(M):
		logAlpha[0][j] = log_startprob[j] + log_emlik[0][j]

	for n in range(1, N):
		for j in range(M):
			logAlpha[n][j] = log_emlik[n][j]
			# building the array of the log sum
			sumArray = []
			for i in range(M):
				sumArray += [logAlpha[n-1][i] + log_transmat[i][j]]
			logAlpha[n][j] +=  tools2.logsumexp(np.array(sumArray))			
	return logAlpha 


lenRemainingZeros = len(concatMatO[0])-len(phoneHMMs[modellist['o'][0]]['startprob'])
piO = phoneHMMs[modellist['o'][0]]['startprob']

piO = np.append(piO, [0 for x in range(lenRemainingZeros)]) 

concatMatO = np.array(concatMatO)

logAlpha = forward(obsloglik, log_inf(piO), log_inf(concatMatO))

# pl.pcolormesh(np.transpose(logAlpha))
# pl.show()


weights = None

def gmmloglik(logAlpha, weights):
	return tools2.logsumexp(np.array(logAlpha[-1]))

print('gmmloglik')
print(gmmloglik(logAlpha, weights) == example['loglik'])

## ----------------
## 4.3
## ---------------


## TODO

def viterbi(log_emlik, log_startprob, log_transmat):
        N = len(log_emlik)
        M = len(log_emlik[0])
        V = [[0 for j in range(M)] for n in range(N)]
        B = [[0 for j in range(M)] for n in range(N)]
        viterbi_path = [0 for i in range(N)]
        for j in range(M):
                V[0][j] = log_startprob[j] + log_emlik[0][j]
        for n in range(1,N):
                for j in range(M):
                        current = V[n-1][0] + log_transmat[0][j]
                        precedent = current
                        for i in range(M):
                                current = max(current, V[n-1][i] + log_transmat[i][j])
                                if (current != precedent):
                                        B[n][j] = i
                                precedent = current
                        V[n][j] = current + log_emlik[n][j]

        viterbi_path[N-1] = np.argmax(V[N-1])
        for i in range(N-2, 0, -1):
                viterbi_path[i] = B[i+1][viterbi_path[i+1]]
        viterbi_loglik = max(V[N-1])
        return (viterbi_loglik, np.array(viterbi_path))
                

vloglik , vpath= example['vloglik']
loglik, path = viterbi(example['obsloglik'], log_inf(piO), log_inf(concatMatO))
#pl.plot(logAlpha)
#pl.plot(path)
#pl.show()
# print(vloglik == loglik)
# print(vpath == path)


## ----------------
## 4.4
## ---------------

def backward(log_emlik, log_startprob, log_transmat):
	N = len(log_emlik)
	M = len(log_emlik[0])

	logBeta = [[0 for x in range(M)] for y in range(N)]

	for i in range(M):
		logBeta[N-1][i] = 0
	for n in range(N-2, -1, -1):
		for i in range(M):
			# building the array of the log sum
			sumArray = []
			for j in range(M):
				sumArray += [log_transmat[i][j] + log_emlik[n+1][j] + logBeta[n+1][j]]
			
			logBeta[n][i] =  tools2.logsumexp(np.array(sumArray))			
	return logBeta 



logBeta = backward(obsloglik, log_inf(piO), log_inf(concatMatO))

#pl.pcolormesh(np.transpose(logBeta))
#pl.show()


## ----------------
## 5.1
## ---------------

def statePosteriors(log_alpha, log_beta):
        N = len(log_alpha)
        M = len(log_alpha[0])
        sum = 0
        y = [[ 0 for j in range(M)] for i in range(N) ]
        sum = tools2.logsumexp(log_alpha[N-1])
        for n in range(N):
                for j in range(M):
                           y[n][j] = log_alpha[n][j] + log_beta[n][j] - sum
        log_gamma = y
        return log_gamma

logGamma =statePosteriors(example['logalpha'], example['logbeta'])
# print("Test of statePosteriors")
# print(logGamma == example['loggamma'])


## ----------------
## 5.2
## ---------------


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
	
	N = len(log_gamma)
	M = len(log_gamma[0])
	D = len(X[0])

	means =  [[0 for j in range(D)] for i in range(M)]
	covars = [[0 for j in range(D)] for i in range(M)]

	for j in range(M):
		sumGammaBottom = 0
		for n in range(N):
			sumGammaBottom += exp(log_gamma[n][j])
		for i in range(D):
			sumGammaTop = 0
			for n in range(N):
				sumGammaTop += exp(log_gamma[n][j])*X[n][i]
				means[j][i] = sumGammaTop/sumGammaBottom
	for j in range(M):
		sumGammaBottom = 0
		for n in range(N):
			sumGammaBottom += exp(log_gamma[n][j])
		for i in range(D):
			sumGammaTop = 0
			for n in range(N):
				xMinusMean = np.subtract(X[n], means[j])
				covarVector = np.multiply(xMinusMean, xMinusMean)
				sumGammaTop += exp(log_gamma[n][j])*covarVector[i]
				covars[j][i] = sumGammaTop/sumGammaBottom

	return means, covars

X = example['lmfcc']
log_gamma = example['loggamma']

updateMeanAndVar(X, log_gamma, varianceFloor=5.0)

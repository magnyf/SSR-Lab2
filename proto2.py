import numpy as np
from tools2 import *

def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
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
    transMat = [hmmmodels[k]['transmat'] for k in namelist]
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

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
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

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """

import numpy as np
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












print(concatHMMs(phoneHMMs, modellist['o']))

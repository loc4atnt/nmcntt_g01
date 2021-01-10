import numpy as np
from sklearn.neighbors import KNeighborsClassifier

KNN = 'KNN'
AVGSAMPLE = 'AVGSAMPLE'

def getAvgSampleTrainDataset(X_raw, y_raw):
	y = np.array(range(10))
	X = np.zeros((10,X_raw.shape[1]))
	for i in y:
		X[i] = np.mean(X_raw[y_raw==i],axis=0)
	return X, y

def getKNNModel(X=None, y=None, n_neighbors=100):
	kNN = KNeighborsClassifier(n_neighbors)
	kNN.fit(X,y)
	return kNN

def getAvgSampleClassifyModel(X_raw=None, y_raw=None):
	X, y = getAvgSampleTrainDataset(X_raw, y_raw)
	kNN = KNeighborsClassifier(1)
	kNN.fit(X,y)
	return kNN

def getModel(mode, X=None, y=None, n_neighbors=100):
	if mode==KNN:
		return getKNNModel(X, y, n_neighbors)
	if mode==AVGSAMPLE:
		return getAvgSampleClassifyModel(X, y)
	return None;
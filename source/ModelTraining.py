import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

KNN = 'KNN'
AVGSAMPLE = 'AVGSAMPLE'

def getAvgSampleTrainDataset(X_raw, y_raw):
	y = np.array(range(10))
	X = np.zeros((10,X_raw.shape[1]))
	for i in y:
		X[i] = np.mean(X_raw[y_raw==i],axis=0)
	return X, y

def getKNNModel(X=None, y=None, n_neighbors=100, isSaved=False, savingPath='knn_model.sav'):
	if isSaved:
		return pickle.load(open(savingPath,'rb'))
	else:
		kNN = KNeighborsClassifier(n_neighbors)
		kNN.fit(X,y)
		# Save model
		pickle.dump(kNN, open(savingPath,'wb'))
		return kNN

def getAvgSampleClassifyModel(X_raw=None, y_raw=None, isSaved=False, savingPath='avg_sample_model.sav'):
	if isSaved:
		return pickle.load(open(savingPath,'rb'))
	else:
		X, y = getAvgSampleTrainDataset(X_raw, y_raw)
		kNN = KNeighborsClassifier(1)
		kNN.fit(X,y)
		# Save model
		pickle.dump(kNN, open(savingPath,'wb'))
		return kNN

def getModel(mode, X=None, y=None, n_neighbors=100, isSaved=False, savingPath='model.sav'):
	if mode==KNN:
		return getKNNModel(X, y, n_neighbors, isSaved, savingPath)
	if mode==AVGSAMPLE:
		return getAvgSampleClassifyModel(X, y, isSaved, savingPath)
	return None;
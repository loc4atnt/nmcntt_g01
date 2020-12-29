import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import DataSet as ds
import Vectorization as vec
import DownSampling as dSample
import Histogram as his
import pickle

HISTOGRAM = 'HISTOGRAM'
VECTORIZATION = 'VECTORIZATION'
DOWNSAMPLING = 'DOWNSAMPLING'

def featureExtract(dataset, tech = HISTOGRAM, option = dSample.AVG, kernel_size = 2):
	if tech == HISTOGRAM:
		return his.getImgArrHisExtraction(dataset)
	if tech == VECTORIZATION:
		return vec.multiImgToVector(dataset)
	if tech == DOWNSAMPLING:
		return dSample.Downsample_List_Arr(dataset, option, kernel_size, isVectorize = True)

def getModel(X=None, y=None, n_neighbors=100, isSaved=False, savingPath='model.sav'):
	if isSaved:
		return pickle.load(open(savingPath,'rb'))
	else:
		kNN = KNeighborsClassifier(n_neighbors)
		kNN.fit(X,y)
		# Save model
		pickle.dump(kNN, open(savingPath,'wb'))
		return kNN

#X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')
X_test = X_test[100:200]
y_test = y_test[100:200]
# Extract test dataset
X_test_extracted = featureExtract(X_test, tech = DOWNSAMPLING)

# Model
# kNN = getModel(featureExtract(X_train, tech = DOWNSAMPLING), y_train)
kNN = getModel(isSaved=True)

print("Du doan tren tap test:")
# Check accuracy
acc = kNN.score(X_test_extracted, y_test)
print("Do chinh xac: %d%%"%(acc*100))
# Predict
print(kNN.predict(X_test_extracted.reshape(X_test_extracted.shape[0],-1)))
print("Ket qua dung:")
print(y_test)
# Show
#ds.showImages(X_test[:10], kNN.predict(X_test_extracted[:10].reshape(10,-1)))
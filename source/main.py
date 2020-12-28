import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import DataSet as ds
import Vectorization as vec
import DownSampling as dSample
import Histogram as his

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

X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')

TRAIN_SIZE = 6000

# Extraction
X_train_extracted = featureExtract(X_train[:TRAIN_SIZE], tech = DOWNSAMPLING)
X_test_extracted = featureExtract(X_test, tech = DOWNSAMPLING)

# Train
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train_extracted,y_train[:TRAIN_SIZE])
# Check accuracy
acc = kNN.score(X_test_extracted, y_test)
print("Do chinh xac: %d%%"%(acc*100))

# Predict
print("Du doan tren tap test:")
print(kNN.predict(X_test_extracted.reshape(X_test_extracted.shape[0],-1)))
print("Ket qua dung:")
print(y_test)
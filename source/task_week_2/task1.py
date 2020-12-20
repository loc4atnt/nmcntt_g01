import numpy as np
# Parameter: img is a numpy array with shape (x,y)
# Return: a numpy array with shape (x*y)
def imgToVector(array2d):
	r = array2d.shape[0]
	c = array2d.shape[1]
	array1d = np.zeros(c*r)
	for i in range(r):
		for j in range(c):
			array1d[i*c+j] = array2d[i, j]
	return array1d # Fake return

from main_Gia_modified import *

X_train, y_train = loadMnist("../data/")
X_test, y_test = loadMnist("../data/", kind='test')

X_train_vectorized = np.zeros ((X_train.shape[0], X_train[0].shape[0]*X_train[0].shape[1]))
for i in range (0, X_train.shape[0]):
        X_train_vectorized [i] = imgToVector(X_train[i])

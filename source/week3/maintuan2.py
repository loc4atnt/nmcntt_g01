import numpy as np

import DataSet as ds
import Vectorization as vec
import DownSampling
import Histogram as his

X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')

#ds.showData(X_train, y_train, X_test, y_test)

# Extraction
#print("Vectorize Image Array.........")
Vec_X_train = vec.multiImgToVector(X_train)
#print("Shape: ",Vec_X_train.shape)
#print(Vec_X_train)

#print("Get Histogram Image Array.........")
Histogram_X_train = his.getImgArrHisExtraction(X_train)
#print("Shape: ",Histogram_X_train.shape)
#print(Histogram_X_train)

#print("Down Sampling Image Array.........")
DownSampl_X_train = DownSampling.Downsample_List_Arr(X_train[:1000], option = DownSampling.AVG,
													 kernel_size = 2, isVectorize = True)
#print("Shape: ",DownSampl_X_train.shape)
#print(DownSampl_X_train)
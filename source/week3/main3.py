import DataSet as ds
import Vectorization as vec
import DownSampling as dsample
import Histogram as his
from sklearn.neighbors import KNeighborsClassifier

HISTOGRAM = 'HISTOGRAM'
VECTORIZATION = 'VECTORIZATION'
DOWNSAMPLING = 'DOWNSAMPLING'

def featureExtract(dataset, tech = HISTOGRAM, option = dsample.AVG, kernel_size = 2):
	if tech == HISTOGRAM:
		return his.getImgArrHisExtraction(dataset)
	if tech == VECTORIZATION:
		return vec.multiImgToVector(dataset)
	if tech == DOWNSAMPLING:
		return dsample.Downsample_List_Arr(dataset, option, kernel_size, isVectorize = True)

X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')


X_train = featureExtract(X_train[:1000], HISTOGRAM)
X_test = featureExtract(X_test, HISTOGRAM)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train[:1000])
#Predict Output
predicted= knn.predict(X_test)
print("y_test: ", predicted)


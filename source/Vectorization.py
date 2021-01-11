import numpy as np
# Parameter: img is a numpy array with shape (x,y)
# Return: a numpy array with shape (x*y)
def imgToVector(array2d):
	array1d = array2d.reshape(array2d.shape[0]*array2d.shape[1])
	return array1d

# Parameter: imgArr is a numpy array with shape (n,x,y) - with n is the number of images
# Return: a numpy array with shape (n, x*y)
	
def multiImgToVector(imgArr):
	arrImgVec = imgArr.reshape ((imgArr.shape[0],imgArr.shape[1]*imgArr.shape[2]))
	return arrImgVec

"""
import DataSet
import time
X_test, y_test = DataSet.loadMnist("./data/", kind='train')
start_time = time.time()
gay2 = multiImgToVector (X_test)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
gay2 = multiImgToVector_test (X_test)
print("--- New %s seconds ---" % (time.time() - start_time))
if np.array_equal(multiImgToVector (X_test), multiImgToVector_test (X_test)):
        print ("EQUAL!!!!")
"""

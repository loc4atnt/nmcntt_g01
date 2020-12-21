import numpy as np
# Parameter: img is a numpy array with shape (x,y)
# Return: a numpy array with shape (x*y)
def imgToVector(array2d):
	array1d = array2d.reshape(array2d.shape[0]*array2d.shape[1])
	return array1d

# Parameter: imgArr is a numpy array with shape (n,x,y) - with n is the number of images
# Return: a numpy array with shape (n, x*y)
def multiImgToVector(imgArr):
	n = imgArr.shape[0]
	arrImgVec = np.zeros((n,imgArr.shape[1]*imgArr.shape[2]))
	for i in range(n):
		arrImgVec[i] = imgToVector(imgArr[i])
	return arrImgVec
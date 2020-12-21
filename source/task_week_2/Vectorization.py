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

# Parameter: imgArr is a numpy array with shape (n,x,y) - with n is the number of images
# Return: a numpy array with shape (n, x*y)
def multiImgToVector(imgArr):
	n = imgArr.shape[0]
	arrImgVec = np.zeros((n,imgArr.shape[1]*imgArr.shape[2]))
	for i in range(n):
		arrImgVec[i] = imgToVector(imgArr[i])
	return arrImgVec
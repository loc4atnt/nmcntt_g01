import numpy as np

import task1
# Parameter: imgArr is a numpy array with shape (n,x,y) - with n is the number of images
# Return: a numpy array with shape (n, x*y)
def multiImgToVector(imgArr):
	n = imgArr.shape[0]
	arrImgVec = np.zeros((n,imgArr.shape[1]*imgArr.shape[2]))
	for i in range(n):
		arrImgVec[i] = imgToVector(imgArr[i])
	return arrImgVec

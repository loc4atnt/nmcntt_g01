import numpy as np

import task1
import DownSampling


# Parameter: imgArr is a numpy array with shape (n,y,y)
#			 n (a number) is size of kernel (n*n)
#			 sampling_kind is a number in SAMPLING KIND (task3.py)
# Return: a numpy array with shape (n, (z=y/n)*(z=y/n))
def getImgArrSamplingExtraction(imgArr, n, sampling_kind):
	z = int(imgArr.shape[1]/n)
	newArrShape = (imgArr.shape[0],z*z)
	samplArr = np.zeros(newArrShape)
	#
	i = 0
	for img in imgArr:
		samplImg = DownSampling.Downsample_Arr(img, sampling_kind, n)
		samplArr[i] = task1.imgToVector(samplImg)
		i += 1
	return samplArr
"""
arr = np.array([[       [2 ,3 ,5 ,2],
 			[1 ,78,23,6],
 			[12,7 ,10,7],
 			[2 ,3 ,5 ,2]],

 		[       [1 ,23,18,12],
 			[9 ,0 ,2 ,3],
 			[4 ,6 ,1 ,3],
 			[9 ,0 ,2 ,3]],

 		[       [12,56,12,5 ],
 			[0 ,12,44,71],
 			[9 ,88,10,34],
 			[9 ,88,10,34]    ]])
print(getImgArrSamplingExtraction(arr, 1, DownSampling.AVG))
print(getImgArrHisExtraction(arr)[0])
print(getImgArrSamplingExtraction(arr, 2, DownSampling.MAX))
print(getImgArrHisExtraction(arr).shape)
"""

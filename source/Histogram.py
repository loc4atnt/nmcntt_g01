import numpy as np

# Parameter: img is a numpy array with shape (x,y)
# Return: a numpy array with shape (256)
def getHistogramExtraction(img):
        # hisArr = np.zeros(256, dtype=int)
        # for row in img:
        #         for pixel in row:
        #                 hisArr[int(pixel)] += 1
        hisArr, bin_edges = np.histogram(img,bins=256)
        return hisArr

# Parameter: imgArr is a numpy array with shape (n,y,y)
# Return: a numpy array with shape (n, 256)
def getImgArrHisExtraction(imgArr):
	hisArr = np.zeros((imgArr.shape[0],256), dtype=int)
	#
	i = 0
	for img in imgArr:
		hisArr[i] = getHistogramExtraction(img)
		i += 1
	return hisArr
import Vectorization as vec
import DownSampling as dSample
import Histogram as his

HISTOGRAM = 'HISTOGRAM'
VECTORIZATION = 'VECTORIZATION'
DOWNSAMPLING = 'DOWNSAMPLING'

def featureExtract(dataset, tech = VECTORIZATION, option = dSample.AVG, kernel_size = 2, isVectorize = True):
	if tech == HISTOGRAM:
		return his.getImgArrHisExtraction(dataset)
	if tech == VECTORIZATION:
		return vec.multiImgToVector(dataset)
	if tech == DOWNSAMPLING:
		return dSample.Downsample_List_Arr(dataset, option, kernel_size, isVectorize)
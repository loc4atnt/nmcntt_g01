import numpy as np

import task3

# Parameter: img is a numpy array with shape (y,y)
#			 n (a number) is size of sampling piece (n*n)
#			 sampling_kind is a number in SAMPLING KIND (task3.py)
# Return: a numpy array with shape (z=y/n,z=y/n)
def getSamplingExtraction(img, n, sampling_kind):
	return np.array([[]]) # Fake return
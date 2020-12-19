import numpy as np

def avgArr(arr):
    return np.mean(arr)
def minArr(arr):
    return np.min(arr)
def maxArr(arr):
    return np.max(arr)

# define --------
AVG = 0
MIN = 1
MAX = 2
#----------------

def getArrSamplingVal(arr, option):
    if option == 0 : return avgArr(arr)
    if option == 1 : return minArr(arr)
    if option == 2 : return maxArr(arr)

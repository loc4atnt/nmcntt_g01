import numpy as np
import task1
import task2
 
def avgArr(arr):
    return np.mean(arr)
def minArr(arr):
    return np.min(arr)
def maxArr(arr):
    return np.max(arr)

def Downsample_Arr(array2d, option):
    array_result = np.zeros ((array2d.shape[0]//2, array2d.shape [0]//2))
    for i in range (0, array2d.shape [0], 2):
        for j in range (0, array2d.shape [0], 2):
            if option == 'AVG' : array_result [i//2, j//2] = avgArr(array2d[i:i+1, j:j+1])
            if option == 'MIN' : array_result [i//2, j//2] = minArr(array2d[i:i+1, j:j+1])
            if option == 'MAX' : array_result [i//2, j//2] = maxArr(array2d[i:i+1, j:j+1])
    return array_result

def Downsample_List_Arr (list_array2d, option):
    array_result = np.zeros ((list_array2d.shape[0], list_array2d.shape[1]//2, list_array2d.shape [1]//2))
    for i in range (0, list_array2d.shape[0]):
        array_result[i] = Downsample_Arr(list_array2d[i], option)
    return array_result

"""
import main
X_test, y_test = main.loadMnist("../data/", kind='test')
gay = Downsample_Arr (X_test[0], 'AVG')
gay2 = Downsample_List_Arr (X_test, 'AVG')
print (gay)
print (gay2[0])
print (gay2.shape)
"""

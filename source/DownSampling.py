import numpy as np
import Vectorization

AVG = 'AVG'
MIN = 'MIN'
MAX = 'MAX'


def Downsample_Arr(array2d, option = 'AVG', kernel_size = 2):
    if option == 'AVG':
        return array2d.reshape ((array2d.shape[0]//kernel_size, kernel_size, array2d.shape [0]//kernel_size, kernel_size)).mean(axis=(1,3))
    elif option == 'MIN':
        return array2d.reshape ((array2d.shape[0]//kernel_size, kernel_size, array2d.shape [0]//kernel_size, kernel_size)).min(axis=(1,3))
    elif option == 'MAX':
        return array2d.reshape ((array2d.shape[0]//kernel_size, kernel_size, array2d.shape [0]//kernel_size, kernel_size)).max(axis=(1,3))
    
#Downsample List of Images
def Downsample_List_Arr (list_array2d, option = 'AVG', kernel_size = 2, isVectorize = True):
    if (list_array2d.shape[1]%kernel_size!=0):
        raise Exception("Error!! Img size must be divisible by kernel_size value")
    if option == 'AVG':
        array_result = list_array2d.reshape ((list_array2d.shape[0], list_array2d.shape[1]//kernel_size, kernel_size, list_array2d.shape [1]//kernel_size, kernel_size)).mean(axis=(2,4))
    elif option == 'MIN':
        array_result = list_array2d.reshape ((list_array2d.shape[0], list_array2d.shape[1]//kernel_size, kernel_size, list_array2d.shape [1]//kernel_size, kernel_size)).min(axis=(2,4))
    elif option == 'MAX':
        array_result = list_array2d.reshape ((list_array2d.shape[0], list_array2d.shape[1]//kernel_size, kernel_size, list_array2d.shape [1]//kernel_size, kernel_size)).max(axis=(2,4))
    return (Vectorization.multiImgToVector(array_result) if isVectorize else array_result)


#Gia: Legacy Code below
"""

def avgArr(arr):
    return np.mean(arr)
def minArr(arr):
    return np.min(arr)
def maxArr(arr):
    return np.max(arr)

def Downsample_Arr_old(array2d, option = 'AVG', kernel_size = 2):
    # if (array2d.shape[0]%kernel_size!=0):
    #     raise Exception("Error!! Img size ",array2d.shape[0]," must be divisible by kernel_size value ",kernel_size,"")    
    array_result = np.zeros ((array2d.shape[0]//kernel_size, array2d.shape [0]//kernel_size))
    range_ = range(0, array2d.shape [0], kernel_size)
    for i in range_:
        for j in range_:
            if option == 'AVG':
                array_result [i//kernel_size, j//kernel_size] = avgArr(array2d[i:i+kernel_size, j:j+kernel_size])
            elif option == 'MIN':
                array_result [i//kernel_size, j//kernel_size] = minArr(array2d[i:i+kernel_size, j:j+kernel_size])
            elif option == 'MAX':
                array_result [i//kernel_size, j//kernel_size] = maxArr(array2d[i:i+kernel_size, j:j+kernel_size])
    return array_result

def Downsample_List_Arr (list_array2d, option = 'AVG', kernel_size = 2, isVectorize = True):
    if (list_array2d.shape[1]%kernel_size!=0):
        raise Exception("Error!! Img size must be divisible by kernel_size value")

    array_result = np.zeros ((list_array2d.shape[0], list_array2d.shape[1]//kernel_size, list_array2d.shape [1]//kernel_size))
    for i in range (0, list_array2d.shape[0]):
        array_result[i] = Downsample_Arr(list_array2d[i], option, kernel_size)

    return (Vectorization.multiImgToVector(array_result) if isVectorize else array_result)
"""

"""
import DataSet
X_test, y_test = DataSet.loadMnist("./data/", kind='train')
#gay = Downsample_Arr (X_test[0], 'MAX', 4)
import time
start_time = time.time()
gay2 = Downsample_List_Arr (X_test, 'AVG', 4)
print("--- %s seconds ---" % (time.time() - start_time))
#print (gay)
print (gay2[3])
print (gay2.shape)


start_time = time.time()
gay2 = Downsample_List_Arr_Test (X_test, 'AVG', 4)
print("--- %s seconds ---" % (time.time() - start_time))
#print (gay)
print (gay2[3])
print (gay2.shape)
"""

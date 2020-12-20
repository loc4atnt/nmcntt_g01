import numpy as np
import task1
import task2
 
def avgArr(arr):
    return np.mean(arr)
def minArr(arr):
    return np.min(arr)
def maxArr(arr):
    return np.max(arr)

#Downsample 1 Image
def Downsample_Arr(array2d, option = 'AVG', strides = 2):
    if (array2d.shape[0]%strides!=0):
        raise Exception("Error!! Img size ",array2d.shape[0]," must be divisible by strides value ",strides,"")
    
    array_result = np.zeros ((array2d.shape[0]//strides, array2d.shape [0]//strides))
    for i in range (0, array2d.shape [0], strides):
        for j in range (0, array2d.shape [0], strides):
            if option == 'AVG' : array_result [i//strides, j//strides] = avgArr(array2d[i:i+strides-1, j:j+strides-1])
            if option == 'MIN' : array_result [i//strides, j//strides] = minArr(array2d[i:i+strides-1, j:j+strides-1])
            if option == 'MAX' : array_result [i//strides, j//strides] = maxArr(array2d[i:i+strides-1, j:j+strides-1])
    return array_result

#Downsample List of Images
def Downsample_List_Arr (list_array2d, option = 'AVG', strides = 2):
    if (list_array2d.shape[1]%strides!=0):
        raise Exception("Error!! Img size must be divisible by strides value")
    
    array_result = np.zeros ((list_array2d.shape[0], list_array2d.shape[1]//strides, list_array2d.shape [1]//strides))
    for i in range (0, list_array2d.shape[0]):
        array_result[i] = Downsample_Arr(list_array2d[i], option, strides)
    return array_result

#Proof of concept, xoa neu nop thay
"""
import main
X_test, y_test = main.loadMnist("../data/", kind='test')
gay = Downsample_Arr (X_test[0], 'AVG', 4)
gay2 = Downsample_List_Arr (X_test, 'AVG', 4)
print (gay)
print (gay2[0])
print (gay2.shape)
"""

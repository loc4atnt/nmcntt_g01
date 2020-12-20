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
def Downsample_Arr(array2d, option = 'AVG', kernel_size = 2):
    if (array2d.shape[0]%kernel_size!=0):
        raise Exception("Error!! Img size ",array2d.shape[0]," must be divisible by kernel_size value ",kernel_size,"")
    
    array_result = np.zeros ((array2d.shape[0]//kernel_size, array2d.shape [0]//kernel_size))
    for i in range (0, array2d.shape [0], kernel_size):
        for j in range (0, array2d.shape [0], kernel_size):
            if option == 'AVG' : array_result [i//kernel_size, j//kernel_size] = avgArr(array2d[i:i+kernel_size, j:j+kernel_size])
            if option == 'MIN' : array_result [i//kernel_size, j//kernel_size] = minArr(array2d[i:i+kernel_size, j:j+kernel_size])
            if option == 'MAX' : array_result [i//kernel_size, j//kernel_size] = maxArr(array2d[i:i+kernel_size, j:j+kernel_size])
    return array_result

#Downsample List of Images
def Downsample_List_Arr (list_array2d, option = 'AVG', kernel_size = 2):
    if (list_array2d.shape[1]%kernel_size!=0):
        raise Exception("Error!! Img size must be divisible by kernel_size value")
    
    array_result = np.zeros ((list_array2d.shape[0], list_array2d.shape[1]//kernel_size, list_array2d.shape [1]//kernel_size))
    for i in range (0, list_array2d.shape[0]):
        array_result[i] = Downsample_Arr(list_array2d[i], option, kernel_size)
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

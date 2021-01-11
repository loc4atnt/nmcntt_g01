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
    else:
    	return Exception("Error!! Not Found %s"%option)
    return (Vectorization.multiImgToVector(array_result) if isVectorize else array_result)

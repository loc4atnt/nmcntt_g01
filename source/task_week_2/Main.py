import numpy as np
import task1, task2, task5
import DownSampling
import Dataset

X_train, y_train = Dataset.loadMnist("../data/")
X_test, y_test = Dataset.loadMnist("../data/", kind='test')

X_train_downed = task5.getImgArrSamplingExtraction(X_train, 2, DownSampling.AVG)



import numpy as np
import random

import DataSet as ds
import FeatureExtraction as fe
import ModelTraining as mt
<<<<<<< HEAD
=======
# import PickleUtil as pu

# import time
# startTime = time.time()
>>>>>>> ef82c377e26d29e6146cc8234f97c72ffd92ddec

# Load bo du lieu MNIST
X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')

# Lay 100 mau Test random
TEST_AMOUNT = 100
rdIdx = random.randrange(10000-TEST_AMOUNT)
X_test = X_test[rdIdx:rdIdx+TEST_AMOUNT]
y_test = y_test[rdIdx:rdIdx+TEST_AMOUNT]

# Extract dataset
# fe.HISTOGRAM
# fe.VECTORIZATION
# fe.DOWNSAMPLING
X_test_extracted = fe.featureExtract(X_test, tech = fe.VECTORIZATION)
X_train_extracted = fe.featureExtract(X_train, tech = fe.VECTORIZATION)

# Model
# mt.AVGSAMPLE
# mt.KNN
<<<<<<< HEAD
model = mt.getModel(mt.AVGSAMPLE, X_train_extracted, y_train)
=======
model = mt.getModel(mt.KNN, X_train_extracted, y_train, n_neighbors=100)
# pu.saveModelToFile(model, "model/m11.sav")
# print(time.time()-startTime)
>>>>>>> ef82c377e26d29e6146cc8234f97c72ffd92ddec

print("Du doan tren mau %d anh test phia tren:"%TEST_AMOUNT)
# Check accuracy
acc = model.score(X_test_extracted, y_test)
print("Do chinh xac: %d%%"%(acc*100))
# Predict
prediction = model.predict(X_test_extracted.reshape(X_test_extracted.shape[0],-1))
print(prediction)
print("Ket qua dung:")
print(y_test)

<<<<<<< HEAD
# Show ngau nhien 10 du doan anh trong tap test
ds.showRandPredictImage(X_test, prediction, y_test)
=======
# ds.showImages(X_test, prediction, y_test)
>>>>>>> ef82c377e26d29e6146cc8234f97c72ffd92ddec

import numpy as np
import random

import DataSet as ds
import FeatureExtraction as fe
import ModelTraining as mt

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
model = mt.getModel(mt.AVGSAMPLE, X_train_extracted, y_train)

print("Du doan tren mau %d anh test phia tren:"%TEST_AMOUNT)
# Check accuracy
acc = model.score(X_test_extracted, y_test)
print("Do chinh xac: %d%%"%(acc*100))
# Predict
prediction = model.predict(X_test_extracted.reshape(X_test_extracted.shape[0],-1))
print(prediction)
print("Ket qua dung:")
print(y_test)

# Show ngau nhien 10 du doan anh trong tap test
ds.showRandPredictImage(X_test, prediction, y_test)
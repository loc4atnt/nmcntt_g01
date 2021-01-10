import numpy as np
import DataSet as ds
import FeatureExtraction as fe
import ModelTraining as mt
import PickleUtil as pu

# import time
# startTime = time.time()

X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')
# X_test = X_test[0:10]
# y_test = y_test[0:10]

# Extract dataset
# fe.HISTOGRAM
# fe.VECTORIZATION
# fe.DOWNSAMPLING
X_test_extracted = fe.featureExtract(X_test, tech = fe.VECTORIZATION)
X_train_extracted = fe.featureExtract(X_train, tech = fe.VECTORIZATION)

# Model
# mt.AVGSAMPLE
# mt.KNN
model = mt.getModel(mt.KNN, X_train_extracted, y_train, n_neighbors=100)
pu.saveModelToFile(model, "model/m11.sav")
# print(time.time()-startTime)

print("Du doan tren tap test:")
# Check accuracy
acc = model.score(X_test_extracted, y_test)
print("Do chinh xac: %d%%"%(acc*100))
# Predict
prediction = model.predict(X_test_extracted.reshape(X_test_extracted.shape[0],-1))
print(prediction)
print("Ket qua dung:")
print(y_test)
# # # Show
# # #ds.showImages(X_test[:10], model.predict(X_test_extracted[:10].reshape(10,-1)))
# # ds.showImage(X_train_avg[6].reshape(14,14))

# print(model.predict(X_test_extracted[5].reshape(1,-1)))
# print("Ket qua dung:")
# print(y_test[5])

# ds.showImages(X_test, prediction, y_test)
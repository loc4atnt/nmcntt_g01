import numpy as np
import DataSet as ds
import FeatureExtraction as fe
import ModelTraining as mt

X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')
# X_test = X_test[300:400]
# y_test = y_test[300:400]

# Extract test dataset
X_test_extracted = fe.featureExtract(X_test, tech = fe.DOWNSAMPLING)

# Model
# model = mt.getModel(mt.AVGSAMPLE,X=fe.featureExtract(X_train, tech = fe.DOWNSAMPLING),y=y_train,savingPath='model/avg_sample_model.sav')
# model = mt.getModel(mt.AVGSAMPLE,isSaved=True,savingPath='model/avg_sample_model.sav')
#
# model = mt.getModel(mt.KNN,X=fe.featureExtract(X_train, tech = fe.DOWNSAMPLING),y=y_train,savingPath='model/knn_model.sav')
model = mt.getModel(mt.KNN,isSaved=True,savingPath='model/knn_model.sav')

print("Du doan tren tap test:")
# Check accuracy
acc = model.score(X_test_extracted, y_test)
print("Do chinh xac: %d%%"%(acc*100))
# Predict
print(model.predict(X_test_extracted.reshape(X_test_extracted.shape[0],-1)))
print("Ket qua dung:")
print(y_test)
# # Show
# #ds.showImages(X_test[:10], model.predict(X_test_extracted[:10].reshape(10,-1)))
# ds.showImage(X_train_avg[6].reshape(14,14))
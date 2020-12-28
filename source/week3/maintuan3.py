import DataSet as ds
import Histogram as his

X_train, y_train = ds.loadMnist("data/", kind='train')
X_test, y_test = ds.loadMnist("data/", kind='test')

from sklearn.neighbors import KNeighborsClassifier

X_train = his.getImgArrHisExtraction(X_train)
X_test = his.getImgArrHisExtraction(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
#Predict Output
predicted= knn.predict(X_test)
accuracy = knn.score(X_test,y_test)
print("Predicted y_test: ", predicted)
print("Accuracy: %d%%"%(accuracy*100))
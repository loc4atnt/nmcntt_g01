import matplotlib.pyplot as plt
import os
import numpy as np
import gzip

def loadMnist(path, kind='train'):
	imgPath = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
	labelPath = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
	with gzip.open(labelPath, 'rb') as labelZip:
		labelZip.read(8)
		buffer = labelZip.read()
		labels = np.frombuffer(buffer, dtype=np.uint8)
	with gzip.open(imgPath, 'rb') as imgZip:
		imgZip.read(16)
		buffer = imgZip.read()
		images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28).astype(np.float64)

	return images, labels

def showData(X_train, y_train, X_test, y_test):
	fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)
	ax = ax.flatten()
	# show train data
	for i in range(10):
		img = X_train[y_train==i][0]
		ax[i].imshow(img, cmap='Greys', interpolation='nearest')
		ax[i].set_title('Train - %d'%i)
	# show random test data
	rdX_test = X_test[np.random.choice(len(X_test), 5, replace=False)]
	for i in range(5):
		img = rdX_test[i]
		ax[10+i].imshow(img, cmap='Greys', interpolation='nearest')
	# setup display
	fig.suptitle('Data Show', fontsize=18);
	ax[10].set_title('Random Test Data:', loc='left')
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	plt.tight_layout()
	plt.show()

#X_train, y_train = loadMnist("data/")
# X_test, y_test = loadMnist("data/", kind='test')
# print("Rows: %d, columns: %d" % (X_train.shape[0], X_train.shape[1]))
# showData(X_train, y_train, X_test, y_test)
#print(X_train[0])

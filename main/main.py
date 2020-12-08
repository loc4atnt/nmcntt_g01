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

def showData(X, y):
	fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
	ax = ax.flatten()
	for i in range(10):
		img = X_train[y_train==i][0]
		ax[i].imshow(img, cmap='Greys', interpolation='nearest')
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	plt.tight_layout()
	plt.show()

X_train, y_train = loadMnist("data/")
print("Rows: %d, columns: %d" % (X_train.shape[0], X_train.shape[1]))
showData(X_train, y_train)
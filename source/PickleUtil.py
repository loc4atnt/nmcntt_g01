import pickle

def loadModelFromFile(savingPath='model.sav'):
	return pickle.load(open(savingPath,'rb'))

def saveModelToFile(model, savingPath='model.sav'):
	pickle.dump(model, open(savingPath,'wb'))
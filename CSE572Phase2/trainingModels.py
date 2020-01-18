import rawDataToFeatureMatrix
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.linear_model import Perceptron as Perc
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle
import random
import os
import warnings


def calculateScores(label, prediction):
	tp,tn,fp,fn = 0,0,0,0
	for i in range(len(prediction)):
		#print(label[i], prediction[i])
		if(label[i] == prediction[i]):
			if(label[i] == 0):
				tn += 1
			else:
				tp += 1
		else:
			if(label[i] ==0):
				fp += 1
			else:
				fn += 1
	if(tp+fp==0 or tp+fn==0):
		return [-1, -1, -1, -1]
	precision, recall = tp/(tp+fp), tp/(tp+fn)
	if(precision+recall==0):
		return [-1, -1, -1, -1]
	F1, accuracy = 2*precision*recall/(precision+recall), (tp+tn)/(tp+tn+fp+fn)
	return [precision, recall, F1, accuracy]

def calculateAverageScores(D):
	precision = {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],'GB': [], 'XGB': [], 'LR': [], 'MLPClassifier': [], 'Perceptron': []}
	recall = {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],
	'GB': [], 'XGB': [], 'LR': [], 'MLPClassifier': [], 'Perceptron': []}
	F1= {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],
	'GB': [], 'XGB': [], 'LR': [], 'MLPClassifier': [], 'Perceptron': []}
	accuracy = {'NeuralNetwork': [], 'RF': [], 'SVM': [], 'KNN': [], 'NB': [],'GB': [], 'XGB': [], 'LR': [], 'MLPClassifier': [], 'Perceptron': []}
	if type(D) is list:
		pass
	else:
		D = [D]
	for d in D:
		#print(d)
		for key,v in d.items():
			precision[key].append(d[key][0])
			recall[key].append(d[key][1])
			F1[key].append(d[key][2])
			accuracy[key].append(d[key][3])
	for key in accuracy.keys():
		precision[key]  = sum(precision[key])/len(precision[key])
		recall[key]  = sum(recall[key])/len(recall[key])
		F1[key]  = sum(F1[key])/len(F1[key])
		accuracy[key]  = sum(accuracy[key])/len(accuracy[key])
		
	return [precision, recall, F1, accuracy]

def MLPClassifier(trainData, trainLable, testData, testLable):
	clf=MLPC(solver='adam', activation='relu', alpha=1e-4,random_state=1,max_iter=200,learning_rate_init=.1)
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/MLPClassifier.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)

def Perceptron(trainData, trainLable, testData, testLable):
	clf = Perc(penalty=None, alpha=0.002, fit_intercept=True, max_iter=10000, verbose=0, tol=1e-4)
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/Perceptron.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)
	
def SVM(trainData, trainLable, testData, testLable):
	clf = svm.SVC(kernel='linear', C=0.5)
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/SVM.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)

def knn(trainData, trainLable, testData, testLable, n_neighbors):
	clf = neighbors.KNeighborsClassifier(n_neighbors)
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/KNN.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)
	
def GNB(trainData, trainLable, testData, testLable):
	clf = GaussianNB()
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/NB.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)
	
def XGBoosting(trainData, trainLable, testData, testLable):
	clf = XGBClassifier()
	clf.fit(trainData, trainLable)
	y_pred = clf.predict(testData)
	predictions = [round(value) for value in y_pred]
	pickle.dump(clf, open('./models/XGB.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)

def gradientBoosting(trainData, trainLable, testData, testLable):
	clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/GB.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)
	
def randomForest(trainData, trainLable, testData, testLable):
	clf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
	clf.fit(trainData, trainLable)
	pickle.dump(clf, open('./models/RF.pkl', 'wb'))
	predict = clf.predict(testData)
	return calculateScores(testLable, predict)

def logisticRegression(trainData, trainLable, testData, testLable):
	clf = LogisticRegression(random_state=0, solver='lbfgs').fit(trainData, trainLable)
	predict = clf.predict(testData)
	pickle.dump(clf, open('./models/LR.pkl', 'wb'))
	return calculateScores(testLable, predict)

def neuralNetwork(trainData, trainLable, testData, testLable):
	model = Sequential()
	model.add(Dense(12, input_dim=len(trainData[0]), activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(np.array(trainData), np.array(trainLable), epochs=200, batch_size=10, verbose=0)
	predict = model.predict_classes(testData)
	pickle.dump(model, open('./models/NN.pkl', 'wb'))
	return calculateScores(testLable, predict)

def trainAllModels(trainData, trainLable, testData, testLable):
	d = {}
	
	nnScore = neuralNetwork(trainData, trainLable, testData, testLable)
	#print("\n\nModel: Precision, Recall, F1, accuracy")
	#print("Neural Network: ",nnScore[3])
	d['NeuralNetwork'] = nnScore
	
	randomForestScore = randomForest(trainData, trainLable, testData, testLable)
	#print("RandomForest: ",randomForestScore[3])
	d['RF'] = randomForestScore
	
	svmScore = SVM(trainData, trainLable, testData, testLable)
	#print("For SVM: ",svmScore[3])
	d['SVM'] = svmScore
	
	knnScore = knn(trainData, trainLable, testData, testLable,8)
	#print("For Knn:",knnScore[3])
	d['KNN'] = knnScore
	
	nbScore = GNB(trainData, trainLable, testData, testLable)
	#print("Naive Bayes: ",nbScore[3])
	d['NB'] = nbScore
	
	gbScore = gradientBoosting(trainData, trainLable, testData, testLable)
	#print("Gradient Boosting: ",gbScore[3])
	d['GB'] = gbScore
	
	xgScore = XGBoosting(trainData, trainLable, testData, testLable)
	#print("XG Boosting: ",xgScore[3])
	d['XGB'] = xgScore
	
	LogRScore = logisticRegression(trainData, trainLable, testData, testLable)
	#print("Logistic Regression: ",LogRScore[3])
	d['LR']=LogRScore
	
	mlpcScore = MLPClassifier(trainData, trainLable, testData, testLable)
	#print("MLPClassifier: ",mlpcScore[3])
	d['MLPClassifier'] = mlpcScore
	
	perceptronScore = Perceptron(trainData, trainLable, testData, testLable)
	#print("Perceptron: ",perceptronScore[3])
	d['Perceptron'] = perceptronScore
	return d
	
def main():
	warnings.filterwarnings("ignore")
	data = []
	label = []
	with open("data.csv",'r') as d:
		with open("label.csv","r") as l:
			for row in d.readlines():
				data.append(list(map(float, row.split(","))))
			for row in l.readlines():
				label.append(list(map(float, row.split(","))))
	
	## k fold validation
	k=4
	data= np.array(data)
	label = np.array(label)
	
	kf = KFold(k, shuffle=True)
	kf.get_n_splits(data)
	df = pd.DataFrame()
	D = []
	for train_index, test_index in kf.split(data):
		trainData = data[train_index]
		trainLable = label[train_index]
		testData = data[test_index]
		testLable = label[test_index]
		D.append(trainAllModels(trainData, trainLable, testData, testLable))
	[avg_precision, avg_recall, avg_f1, avg_accuracy]= calculateAverageScores(D)
	print("%-16s | %8s | %8s | %8s | %8s"%('','Precision','Recall','F1','Accuracy'))
	for key,val in avg_precision.items():
		  print("%16s | %5f  | %5f | %5f | %5f"%(key, avg_precision[key], avg_recall[key], avg_f1[key], avg_accuracy[key]))

	
	
if __name__ == "__main__":

	rawDataToFeatureMatrix.main()
	main()
	
	
	



	
	
	
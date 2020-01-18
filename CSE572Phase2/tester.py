### Tester.py  #####
from rawDataToFeatureMatrix import getFeatureVector
from trainingModels import calculateScores
import pandas as pd
import json
import numpy as np
import pickle
import os


# import your function from your file



models = ['MLP_Classifier','Perceptron','SV_Classifier','KNN','NaiveBayes','XG_Boosting', 'GradientBoosting','RandomForest','LogReg','NeuralNetwork']
models_path = ['./models/MLPClassifier.pkl',
			'./models/Perceptron.pkl',
			'./models/SVM.pkl',
			'./models/KNN.pkl',
			'./models/NB.pkl',
			'./models/XGB.pkl',
			'./models/GB.pkl',
			'./models/RF.pkl',
			'./models/LR.pkl',
			'./models/NN.pkl']

#def main():
def tester(inputFileName):
	data = []
	with open(inputFileName, 'r') as f:
		lines = f.readlines()
		for line in lines:
			temp = []
			for elem in line.split(","):
				if(elem.replace('.', '', 1).isdigit()):
					temp.append(int(elem))					
			data.append(temp)
	
	data = getFeatureVector(data)
	
	#applying PCA
	pca = pickle.load(open("./models/pca.pkl", 'rb'))
	data=np.array(data)
	data.reshape(1,-1)
	featureMatrix = pca.transform(data)
	
	for modelName,modelPath in zip(models,models_path):
		model = pickle.load(open(modelPath, 'rb'))
		print("For {}".format(modelName))
		predict = model.predict(featureMatrix)
		print("The output will be saved in the Output Folder!")
		# saving predictions to file 
		if(not os.path.isdir("./Output")):
			os.mkdir("./Output")
		outputFileName = "./Output/{}_output.csv".format(modelName)
		with open(outputFileName, 'w') as f:
			f.writelines("%s\n"% pre for pre in predict)
		
if __name__ == '__main__':
	import sys
	if(len(sys.argv)>1):
		testFile = sys.argv[1]
		tester(testFile)
	else:
		print("pass the file name along wth argument!")

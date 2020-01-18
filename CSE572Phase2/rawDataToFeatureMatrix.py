import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score,classification_report
import scipy
import warnings
warnings.filterwarnings("ignore")

def rms(data):
	####   Root Mean Square  #####
	rms = []
	for i in range(len(data)):
		rms.append(np.sqrt(np.mean(np.square(data[i]))))
	return rms

def fft(data):
	##### FFT   #####
	FFT = []
	for i in range(len(data)):
		FFT.append(np.fft.irfft(data[i], n=2))
	return FFT

def windowedMean(data):
	#### Windowed mean ####
	win_mean = []
	for i in range(len(data)):
		temp = []
		if(len(data[i])<4):
			data[i].extend([0]*(4-len(data[i])))
		n = math.ceil(len(data[i])/4)
		parts = [data[i][j: j+n] for j in range(0, len(data[i]), n)]
		for p in parts:
			mean = [np.mean(p)]
			temp.extend(mean)
		win_mean.append(temp)
	return win_mean

def rangeLargerThanStandardDeviation(data):
	####  Range larger than standard deviation  #####
	variance = []
	K= [0.2, 0.25, 0.3, 0.35, 0.4]
	for i in range(len(data)):
		temp = []
		s = np.std(data[i])
		for k in K:
			if(s> (max(data[i]) - min(data[i]))*k):
				temp.extend([1])
			else:
				temp.extend([0])
		variance.append(temp)
	return variance

def polynoimalFit(data):
	# polynomial parameters
	polyfit = []
	for i in data:
		y = list(range(1, len(i)+1))
		z = np.polyfit(y,i, deg=4)
		polyfit.append(np.poly1d(z).c)
	return polyfit
	

def maxMinusMeanLargerThanStandardDeviation(data):
    ####  Range larger than standard deviation  #####
    variance = []
    K = [0.2, 0.25, 0.3, 0.35, 0.4]
    for i in range(len(data)):
        temp = []
        s = np.std(data[i])
        for k in K:
            if (s > (max(data[i]) - np.mean(data[i])) * k):
                temp.extend([1])
            else:
                temp.extend([0])
        variance.append(temp)
        # print(variance)
    return variance


def maxMinusMeanLarger(data):
    ####  Range larger than standard deviation  #####
    variance = []
    for i in range(len(data)):
        if min(data[i]) == 0:
            minValue = 1
        else:
            minValue = min(data[i])
        variance.append(100 * (max(data[i]) - minValue) / minValue)
    return variance
	
### reading CSV to get the data
def myReadCSV(path):
	data = []    
	with open(path, 'r') as f:
		for line in f.readlines():
			d = line.split(",")
			i, mean = 0, 0 
			temp = []
			flag = 1
			while(i<len(d)):
				if(d[i] == "NaN" or d[i]=="NaN\n"):
					flag = -1
					break
					temp.append(mean)
				else:
					temp.append(int(d[i]))
					mean += (int(d[i])-mean)/(i+1)
				i += 1
			if(flag!=-1):
				data.append(temp)
	return data
	
def getFeatureVector(data):
	final_new = []
	RMS = rms(data)
	FFT = fft(data)
	win_mean = windowedMean(data)
	variance = rangeLargerThanStandardDeviation(data)
	
	variance2 = maxMinusMeanLargerThanStandardDeviation(data)
	variance3 = maxMinusMeanLarger(data)
	
	polyFit = polynoimalFit(data)
	
	len_fft = len(FFT[0])
	len_win_mean = len(win_mean[0])
	len_var = len(variance[0])
	len_var2 = len(variance2[0])

	
	for i in range(len(FFT)):
		temp = [] 
		# rms, fft0, fft1, movAvg0, movAvg1, movAvg2, movAvg3, varaince, polynoimalFit
		temp.append(RMS[i])   # rms
		temp.extend([variance3[i]])  # variance 3			
		
		for j in range(len_fft):  ## fft
			temp.append(FFT[i][j])

		for j in range(len(win_mean[i])):  # moving average
			temp.append(win_mean[i][j])

		for j in range(len_var):  # variance2
			temp.append(variance[i][j])
		
		for j in range(len_var2):  # variance2
			temp.append(variance2[i][j])
		
		temp.extend(polyFit[i])
		
		final_new.append(temp)
	return final_new
			
def main():	
	## reading Meal Data !
	mealData = []
	for i in range(1,6):
		mealData.extend(myReadCSV("Data/mealData"+str(i)+".csv"))

	noMealData = []
	for i in range(1,6):
		noMealData.extend(myReadCSV("Data/Nomeal"+str(i)+".csv"))    
	
	Data = []
	label = []
	for i in range(len(mealData)):
		Data.append(mealData[i])
		label.append(1)
	for i in range(len(noMealData)):
		Data.append(noMealData[i])
		label.append(0)
	topFeatures = 6
	data = np.array(getFeatureVector(Data))
	#Applying PCA and saving model
	pca = PCA(n_components=topFeatures)
	pca = pca.fit(data)
	updatedFeatureVector = pca.transform(data)

	#from joblib import dump
	import pickle
	#dump(pca, './models/pca.joblib') 
	pickle.dump(pca, open('./models/pca.pkl', 'wb'))
		
	import random
	c = list(zip(updatedFeatureVector, label))
	random.shuffle(c)
	data, label = zip(*c)
	data, label = np.array(data), np.array(label)
	
	np.savetxt("data.csv", data, delimiter=",", fmt='%s')
	np.savetxt("label.csv", label, delimiter=",", fmt='%s')
	
if __name__ == '__main__':
	main()

		
		
		
		
		
		
		
		
		
		
		
		
		
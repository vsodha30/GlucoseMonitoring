------- Readme --------
Files:
1. rawDataToFeatureMatrix.py: Reads the CGM data of all patients from "Data" Folder and prepares two files (data, label)
2. trainingModels.py: Takes data from the files created in previous step and applies k Fold(k=4) cross validation and results scores of all algorithms used.
3. tester.py: to test the saved models.

run as: "python tester.py inputFileName.csv"
While running on console, give inputFileName.csv as parameter, the output will be saved in output folder. 


Folder info:
Data: folder where all data is stored
models: folder where all trained models are stored
Output: folder to save the output files


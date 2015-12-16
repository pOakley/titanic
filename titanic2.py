import numpy as np
import pandas as pd
import random as rd
import csv as csv
import pylab as P
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import math

def main():

	trainFile = "train.csv"
	testFile = "test.csv"


	trainData = getData(trainFile)
	cleanTrainData = cleanData(trainData)
	forest = evalData(cleanTrainData)

	testData = getData(testFile)
	testIDs = testData.PassengerId.values
	cleanTestData = cleanData(testData)

	predictResults(cleanTestData, forest, testIDs,outputFile="predictions.csv")

	print('Done.')

def getData(theFile):

	allData = pd.read_csv(theFile)

	return(allData)


def cleanData(theData):
# VARIABLE DESCRIPTIONS:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)


	cleanData = theData

	cleanData = cleanData.drop('Name', axis=1)

	cleanData['Sex'] = cleanData['Sex'].map({'female': 0, 'male': 1}).astype(int)

	#These are roughly ranked in order of importance I think	
	childMedianAge = cleanData['Age'][cleanData['Age'] < 18].median()
	cabinMedianAge = cleanData['Age'][cleanData['Cabin'].isnull() == False].median()
	adultMedianAge = cleanData['Age'][cleanData['Age'] > 18].median()
	allMedianAge = cleanData.Age.dropna().median()

	cleanData['Cabin'][cleanData['Cabin'].isnull() == True] = "-"




	for index, entry in cleanData[cleanData['Age'].isnull()==True].iterrows():

		if entry.SibSp > 1:
			cleanData.loc[index,'Age'] = childMedianAge
		elif entry['Cabin'] != "-":
			cleanData.loc[index,'Age'] = cabinMedianAge
		elif entry.Parch > 2:
			cleanData.loc[index,'Age'] = adultMedianAge
		else:
			cleanData.loc[index,'Age'] = allMedianAge



	cleanData = cleanData.drop('Ticket', axis=1)

	if len(cleanData.Fare[cleanData.Fare.isnull()]) > 0:
		replaceFare = np.zeros(3)
		for k in range(0,3):
			replaceFare[k] = cleanData['Fare'][cleanData['Pclass']==(k+1)].dropna().median()
			cleanData['Fare'][ (cleanData['Pclass']==(k+1)) & (cleanData['Fare'].isnull())] = replaceFare[k]






	cleanData = cleanData.drop('Cabin', axis=1)

	cleanData['Embarked'][cleanData.Embarked.isnull()] = cleanData.Embarked.dropna().mode().values
	cleanData['Embarked'] = cleanData['Embarked'].map( {'C': 0, 'Q': 1,'S': 2} ).astype(int)

	cleanData = cleanData.drop(['PassengerId'], axis=1)

	print(cleanData.columns)

	return(cleanData.values)


def evalData(theData):
	print('Training...')
	forest = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state = 42, max_features="auto",min_samples_leaf=5)
	forestFit = forest.fit( theData[0::,1::], theData[0::,0] )
	scores = cross_validation.cross_val_score(forest,theData[0::,1::],theData[0::,0],cv=3)
	print(scores.mean())
	#print(roc_auc_score(theData[0::,0],forest.oob_prediction_))
	return(forestFit)


def predictResults(theData, forest, theIDs,outputFile):
	print('Predicting...')
	output = forest.predict(theData).astype(int)

	predictionsFile = open(outputFile, "wt")
	open_file_object = csv.writer(predictionsFile)
	open_file_object.writerow(['PassengerId','Survived'])
	open_file_object.writerows(zip(theIDs, output))
	predictionsFile.close()




if __name__ == "__main__":

	main()





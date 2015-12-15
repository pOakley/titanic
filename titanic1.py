import numpy as np
import pandas as pd
import random as rd
import csv as csv

from sklearn.ensemble import RandomForestClassifier 


inputFile = "train.csv"
testData = "test.csv"


allData = pd.read_csv(inputFile)
totalPassengers = allData.shape[0]


firstClassPercentage = sum(allData[allData['Pclass']==1].Survived) / allData[allData['Pclass']==1].shape[0]
secondClassPercentage = sum(allData[allData['Pclass']==2].Survived) / allData[allData['Pclass']==2].shape[0]
thirdClassPercentage = sum(allData[allData['Pclass']==3].Survived) / allData[allData['Pclass']==3].shape[0]

womenPercentage = sum(allData[allData['Sex']=="female"].Survived) / allData[allData['Sex']=="female"].shape[0]
menPercentage = sum(allData[allData['Sex']=="male"].Survived) / allData[allData['Sex']=="male"].shape[0]

agePercentage = np.zeros(8)
ages = np.arange(8)*10+10
for ageMax in range(0,8):
	agePercentage[ageMax] = sum(allData[allData['Age']<ages[ageMax]].Survived) / allData[allData['Age']<ages[ageMax]].shape[0]

siblingSpousePercentage = sum(allData[allData['SibSp'] > 0].Survived) / allData[allData['SibSp']>0].shape[0]
parentsKidsPercentage = sum(allData[allData['Parch'] > 0].Survived) / allData[allData['Parch']>0].shape[0]

fares = np.arange(52)*10+10
farePercentage = np.zeros(52)

for fareMax in range(np.size(fares)):
	farePercentage[fareMax]  = sum(allData[allData['Fare']<fares[fareMax]].Survived) / allData[allData['Fare']<fares[fareMax]].shape[0]


cherbourgPercentage = sum(allData[allData['Embarked']=="C"].Survived) / allData[allData['Embarked']=="C"].shape[0]
queenstownPercentage = sum(allData[allData['Embarked']=="Q"].Survived) / allData[allData['Embarked']=="Q"].shape[0]
southhamptonPercentage = sum(allData[allData['Embarked']=="S"].Survived) / allData[allData['Embarked']=="S"].shape[0]

print("First", firstClassPercentage)
print("Second", secondClassPercentage)
print("Third", thirdClassPercentage)
print("Women", womenPercentage)
print("Men",	menPercentage)
print("Age", agePercentage)
print("Siblings or Spouse", siblingSpousePercentage)
print("Parents or Kids",	parentsKidsPercentage)
print("Fare", farePercentage)
print("Cherbough",	cherbourgPercentage)
print("Queenstown",	queenstownPercentage)
print("Southampton", southhamptonPercentage)

allData['Gender'] = allData['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
allData['Embarked'][61]='C'
allData['Embarked'][829]='C'
allData['Embarked'] = allData['Embarked'].map( {'C': 0, 'Q': 1,'S': 2} ).astype(int)

nanSpots = np.where(allData['Age'].isnull()==True)

for spot in nanSpots:
	allData['Age'][spot]= 30

lessData = allData.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
lessData = lessData.dropna()

lessData.info()

fakeTestData = lessData.drop(['Survived'], axis = 1)

trainData = lessData.values
fakeTest = fakeTestData.values


# TEST DATA
test_df = pd.read_csv(testData, header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
Ports = list(enumerate(np.unique(test_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


test_data = test_df.values




print('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( trainData[0::,1::], trainData[0::,0] )

print('Predicting...')
output = forest.predict(test_data).astype(int)


predictions_file = open("try1.csv", "wt")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['PassengerId','Survived'])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done.')


# results = np.zeros(totalPassengers)
# prob = results
# rd.seed()
# success = 0

# for testCase in range(0,totalPassengers):
# 	prob[testCase] = 0.499
# 	if allData.Sex[testCase]=="female":
# 		prob[testCase] = womenPercentage
# 	else:
# 		prob[testCase] = menPercentage

# 	if (allData.Pclass[testCase] == 1):
# 		prob[testCase] *= firstClassPercentage
# 	elif (allData.Pclass[testCase] == 3):
# 		prob[testCase] *= thirdClassPercentage


# prob = np.divide(prob,np.amax(prob))

# for k in range(0,totalPassengers):
# 	if prob[k]>0.5:
# 		results[k] = 1
# 	if results[k] == allData.Survived[k]:
# 		success += 1

# print(prob)
# print(success)
# print(success / totalPassengers)



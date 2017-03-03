## Linear model skeleton for Problem 3.
## version 0.1

"""
## Overview for problem 3 (Not sure if i understood it correctly)
1. Create a logistic regression model for CTR estimation (Cos predicting for click=1 or 0, so use logRegression)
    y ~ x
    where y = click
    where x = a list of variables we choose
    TODO:
        Fixed - disregard accuracy for now (Means will not use validation data... Will fix/update in next version)
        - Still cannot handle alphabetical variables e.g. useragent .... Will fix/update in next version.
        Fixed - The data has too many click=0, trained model is skewed. To refine train model
2. CTR estimation
    - Estimate click for every record in the test set using the model above
    - Compute pCTR = sumofclicks/alltestrecords
    TODO:
        - Assume all same advertiser for now (Will fix in next version)

3. Compute bid = base_bid x pCTR/avgCTR
    - A bit lost here, what's avgCTR ??

"""

import numpy as np
from patsy import patsy
from sklearn.linear_model import LogisticRegression
import ipinyouReader as ipinyouReader
from ipinyouWriter import ResultWriter as ResultWriter
from sklearn import metrics

# #List of column names. To be copy and pasted (as needed) in the formula for logistic regression
# click='click'
# weekday='weekday'
# hour='hour'
# bidid='bidid'
# logtype='logtype'
# userid='userid'
# useragent='useragent'
# IP='IP'
# region='region'
# city='city'
# adexchange='adexchange'
# domain='domain'
# url='url'
# urlid='urlid'
# slotid='slotid'
# slotwidth='slotwidth'
# slotheight='slotheight'
# slotvisibility='slotvisibility'
# slotformat='slotformat'
# slotprice='slotprice'
# creative='creative'
# bidprice='bidprice'
# payprice='payprice'
# keypage='keypage'
# advertiser='advertiser'
# usertag='usertag'




regressionFormulaY='click'
regressionFormulaX='region + city +slotwidth + slotheight + slotprice'
trainset="../dataset/trainPruned.csv"
validationset="../dataset/validationPruned.csv"
testset="../dataset/test.csv"


##########################
## Modelling and training


# load dataset
print("Reading dataset...")
trainDF = ipinyouReader.ipinyouReader(trainset).getDataFrame()

print("Setting up Y and X for logistic regression")
yTrain, xTrain =patsy.dmatrices(regressionFormulaY + ' ~ ' + regressionFormulaX,trainDF, return_type="dataframe")
print((xTrain.columns))
print ("No of features in input matrix: %d" % len(xTrain.columns))

# flatten y into a 1-D array
print("Flatten y into 1-D array")
yTrain = np.ravel(yTrain)

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
print("Training Model...")
model = model.fit(xTrain, yTrain)   #Loss function:liblinear

# check the accuracy on the training set
print("\n\nTraining acccuracy: %5.3f" % model.score(xTrain, yTrain))


########################
## Prediction of validation set (Clicks)
print("Reading validation set")
validateDF = ipinyouReader.ipinyouReader(validationset).getDataFrame()


print("Setting up X Y validation for prediction")
yValidate, xValidate =patsy.dmatrices(regressionFormulaY + ' ~ ' + regressionFormulaX,validateDF, return_type="dataframe")
print ("No of features in input matrix: %d" % len(xValidate.columns))

# predict click labels for the validation set
print("Predicting validation set...")
predicted = model.predict(xValidate) #0.5 prob threshold
print("Writing to csv")
valPredictionWriter=ResultWriter()
valPredictionWriter.writeResult(filename="predictValidate.csv", data=predicted)
print ("\n\nPrediction acc on validation set: %f5.3" % metrics.accuracy_score(yValidate, predicted))

########################
## Prediction of test set (Clicks)
print("Reading test set")
testDF = ipinyouReader.ipinyouReader(testset).getDataFrame()

print("Setting up X test for prediction")
xTest =patsy.dmatrix(regressionFormulaX,testDF, return_type="dataframe")
print ("No of features in input matrix: %d" % len(xValidate.columns))

# predict click labels for the test set
print("Predicting test set...")
predicted = model.predict(xTest) #0.5 prob threshold
print("Writing to csv")
testPredictionWriter=ResultWriter()
testPredictionWriter.writeResult(filename="predictTest.csv", data=predicted)





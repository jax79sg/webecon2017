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

from BidModels import BidModelInterface

class LogisticRegressionBidModel(BidModelInterface):
    _regressionFormulaY =''
    _regressionFormulaX =''
    _model=None
    def __init__(self, regressionFormulaY='click', regressionFormulaX='weekday + hour + region + city + adexchange +slotwidth + slotheight + slotprice + advertiser'):
        self._regressionFormulaY=regressionFormulaY
        self._regressionFormulaX = regressionFormulaX
        self.defaultBid = 0

    def getBidPrice(self, oneBidRequest):
        print("Setting up X test for prediction")
        xTest = patsy.dmatrix(self._regressionFormulaX, oneBidRequest, return_type="dataframe")

        # predict click labels for the bid request set
        print("Predicting test set...")
        predicted = self._model.predict(xTest)  # 0.5 prob threshold
        print("Writing to csv")
        testPredictionWriter = ResultWriter()
        testPredictionWriter.writeResult(filename="predictTest.csv", data=predicted)

        #Compute the bid price based on prediction outcome


        return [oneBidRequest[2], self.defaultBid]

    def trainModel(self, allTrainData):
        print("Setting up Y and X for logistic regression")
        yTrain, xTrain = patsy.dmatrices(self._regressionFormulaY + ' ~ ' + self._regressionFormulaX, allTrainData, return_type="dataframe")
        print((xTrain.columns))
        print("No of features in input matrix: %d" % len(xTrain.columns))

        # flatten y into a 1-D array
        print("Flatten y into 1-D array")
        yTrain = np.ravel(yTrain)

        # instantiate a logistic regression model, and fit with X and y
        self._model = LogisticRegression()
        print("Training Model...")
        self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear

        # check the accuracy on the training set
        print("\n\nTraining acccuracy: %5.3f" % self._model.score(xTrain, yTrain))

    def validateModel(self, allValidateData):
        if(self._model!=None):
            print("Setting up X Y validation for prediction")
            yValidate, xValidate = patsy.dmatrices(self._regressionFormulaY + ' ~ ' + self._regressionFormulaX, allValidateData, return_type="dataframe")
            print("No of features in input matrix: %d" % len(xValidate.columns))

            # predict click labels for the validation set
            print("Predicting validation set...")
            predicted = self._model.predict(xValidate)  # 0.5 prob threshold
            print("Writing to csv")
            valPredictionWriter = ResultWriter()
            valPredictionWriter.writeResult(filename="predictValidate.csv", data=predicted)
            print("\n\nPrediction acc on validation set: %f5.3" % metrics.accuracy_score(yValidate, predicted))
        else:
            print("Error: No model was trained in this instance....")


# load dataset
print("Reading dataset...")
trainset = "../dataset/train.csv"
validationset = "../dataset/validation.csv"
testset = "../dataset/test.csv"
# trainDF = ipinyouReader.ipinyouReader(trainset).getDataFrame()
reader_encoded = ipinyouReader.ipinyouReaderWithEncoding()
trainDF, validateDF, testDF, lookupDict = reader_encoded.getTrainValidationTestDD(trainset, validationset, testset)
print("Training dataset...")
lrBidModel=LogisticRegressionBidModel()
lrBidModel.trainModel(trainDF)
# print("trainDF.info(): ", trainDF.info())


########################
## Prediction of validation set (Clicks)
# print("Reading validation set")
# validateDF = ipinyouReader.ipinyouReader(validationset).getDataFrame()




########################
## Prediction of test set (Clicks)
# print("Reading test set")
# testDF = ipinyouReader.ipinyouReader(testset).getDataFrame()







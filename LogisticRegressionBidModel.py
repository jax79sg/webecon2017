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
from sklearn.grid_search import GridSearchCV

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
        self._model = LogisticRegression(C=0.1)
        print("Training Model...")
        self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear

        # check the accuracy on the training set
        print("\n\nTraining acccuracy: %5.3f" % self._model.score(xTrain, yTrain))


    def gridSearchandCrossValidate(self, allTrainData):
        print("Setting up Y and X for logistic regression")
        yTrain, xTrain = patsy.dmatrices(self._regressionFormulaY + ' ~ ' + self._regressionFormulaX, allTrainData,
                                         return_type="dataframe")
        print((xTrain.columns))
        print("No of features in input matrix: %d" % len(xTrain.columns))

        # flatten y into a 1-D array
        print("Flatten y into 1-D array")
        yTrain = np.ravel(yTrain)

        # LogisticRegression(penalty='l2',
        #                    dual=False,
        #                    tol=0.0001,
        #                    C=1.0,
        #                    fit_intercept=True,
        #                    intercept_scaling=1,
        #                    class_weight=None,
        #                    random_state=None,
        #                    solver='liblinear',
        #                    max_iter=100,
        #                    multi_class='ovr',
        #                    verbose=0,
        #                    warm_start=False,
        #                    n_jobs=1)

        ## Setup Grid Search parameter

        param_grid = [{
                          'solver': ['liblinear'],
                          'C': [0.095, 0.1, 0.105],
                          'class_weight':[None],  # None is better
                          'penalty': ['l2', 'l1'],
                      }
                        ,
                      {
                          'solver': ['newton-cg', 'lbfgs', 'sag'],
                          'C': [0.09, 0.1, 0.2, 1.0],
                          'max_iter':[50000],
                          'class_weight': [None, 'Balanced'],  # None is better
                          'penalty': ['l2'],
                      }
                      ]

        optimized_LR = GridSearchCV(LogisticRegression(),
                                     param_grid=param_grid,
                                     scoring='accuracy',
                                     cv=5,
                                     n_jobs=-1,
                                     error_score='raise')
        self._model = optimized_LR.fit(xTrain, yTrain)

        scores = optimized_LR.grid_scores_
        # print(type(scores))
        for i in range(len(scores)):
            print(optimized_LR.grid_scores_[i])

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


if __name__ == "__main__":
    # load datasets
    print("Reading dataset...")
    trainset = "../dataset/train_cleaned_prune.csv"
    validationset = "../dataset/validation_cleaned_prune.csv"
    testset = "../dataset/test.csv"
    # trainDF = ipinyouReader.ipinyouReader(trainset).getDataFrame()
    reader_encoded = ipinyouReader.ipinyouReaderWithEncoding()
    trainDF, validateDF, testDF, lookupDict = reader_encoded.getTrainValidationTestDD(trainset, validationset, testset)
    print("Training dataset...")
    lrBidModel=LogisticRegressionBidModel()
    lrBidModel.gridSearchandCrossValidate(trainDF)
    lrBidModel.trainModel(trainDF)
    # print("trainDF.info(): ", trainDF.info())








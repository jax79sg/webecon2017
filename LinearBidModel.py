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
    - Compute pCTR = prob(click=1)

3. Compute bid = base_bid x pCTR/avgCTR
where base_bid assumed budget, avgCTR assumed CTR of training set.

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
"""

import numpy as np
from patsy import patsy
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import ipinyouReader as ipinyouReader
from ipinyouWriter import ResultWriter as ResultWriter
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from UserException import ModelNotTrainedException
from sklearn.externals import joblib
import datetime
import pandas as pd

from BidModels import BidModelInterface

class LinearBidModel(BidModelInterface):
    _regressionFormulaY =''
    _regressionFormulaX =''
    _model=None
    _cBudget=0
    _avgCTR=0
    _modelType=None


    def __init__(self, regressionFormulaY='click', regressionFormulaX='weekday + hour + region + city + adexchange +slotwidth + slotheight + slotprice + advertiser',cBudget=25000*1000, avgCTR=0.000754533880574, modelType="logisticregression"):
        """

        :param regressionFormulaY:
        :param regressionFormulaX:
        :param cBudget:
        :param avgCTR:
        :param modelType: Options ['logisticregression', 'sgdclassifier']
        """
        self._regressionFormulaY=regressionFormulaY
        self._regressionFormulaX = regressionFormulaX
        self._defaultBid = 0
        self._cBudget=cBudget
        self._avgCTR=avgCTR
        self._modelType=modelType

    def __computeBidPrice(self, pCTR=None):
        """
        The default computation to compute bid price
        The implemented model should have its own ways to gather the necessary parameters as follows
        :param basebid:Using the budget in this case
        :param pCTR: Compute the probability that click=1 for that bidrequest
        :param avgCTR: Consider this as the avgCTR for the training set
        :return: bid
        """
        bid=self._cBudget*(pCTR/self._avgCTR)
        return bid

    def __predictClickOneProb(self,testDF):
        """
        Perform prediction for click label.
        Take the output of click=1 probability as the CTR.
        :param oneBidRequest:
        :return:
        """

        print("Setting up X test for prediction")
        xTest = patsy.dmatrix(self._regressionFormulaX, testDF, return_type="dataframe")

        # predict click labels for the test set
        print("Predicting test set...")
        predictedClickOneProb = self._model.predict_proba(xTest)

        return predictedClickOneProb[:,1]


    def getBidPrice(self, allBidRequest):
        """
        1. Predict click=1 prob for entire test/validation set
            Considered as pCTR for each impression
        2. Use the bid=base_price*(pCTR/avgCTR) formula
        :param oneBidRequest:
        :return:
        """

        if(self._model==None):
            raise ModelNotTrainedException("Model must be trained prior to prediction!")


        #Compute the CTR of this BidRequest
        pCTR=self.__predictClickOneProb(allBidRequest)
        print("General sensing of pCTR ranges")
        print(pCTR)

        #Compute the bid price
        bids = np.apply_along_axis(self.__computeBidPrice, axis=0, arr=pCTR)
        print("General sensing of bids ranges")
        print(bids)

        #Extract the corresponding bidid
        allBidRequestMatrix=allBidRequest.as_matrix(columns=['bidid'])

        #Merging bidid and bids into a table (Needed for eval)
        bidid_bids=np.column_stack((allBidRequestMatrix, bids))

        bids = pd.DataFrame(bidid_bids, columns=['bidid', 'bidprice'])
        return bids


    def trainModel(self, allTrainData, retrain=True, modelFile=None):
        """
        Train model using Logistic Regression for Click against a set of features
        Trained model will be saved to disk (No need retrain/reload training data in future if not required during program rerun)
        :param allTrainData:
        :param retrain: If False, will load self._modelFile instead of training the dataset.
        :return:
        """
        self._modelFile=modelFile
        # instantiate a logistic regression model
        if(self._modelType=='logisticregression'):
            print("Logistic Regression will be used for training")
            self._model = LogisticRegression(C=0.1)
        elif (self._modelType=='sgdclassifier'):
            print("SGD classifier will be used for training")
            #alpha=0.01 (big) to make sure the learning rate is bigger as well when using learning_rate='optimal'.
            #loss='log', to get probabilitic estimate

            self._model = SGDClassifier(alpha=0.01, average=False, class_weight=None, epsilon=0.1,
        eta0=10, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='log', n_iter=100000, n_jobs=-1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=1, warm_start=False)
        else:
            print("Unrecognised modelType: Logistic Regression defaulted training")
            self._model = LogisticRegression()


        if (retrain):
            print("Setting up Y and X for logistic regression")
            print(datetime.datetime.now())
            yTrain, xTrain = patsy.dmatrices(self._regressionFormulaY + ' ~ ' + self._regressionFormulaX, allTrainData, return_type="dataframe")
            print("No of features in input matrix: %d" % len(xTrain.columns))

            # flatten y into a 1-D array
            print("Flatten y into 1-D array")
            print(datetime.datetime.now())
            yTrain = np.ravel(yTrain)


            print("Training Model...")
            print(datetime.datetime.now())

            self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear
            super(LinearBidModel, self).saveModel(self._model, self._modelFile)
            # check the accuracy on the training set
            print("\n\nTraining acccuracy: %5.3f" % self._model.score(xTrain, yTrain))
        else:
            self._model=super(LinearBidModel, self).loadSavedModel(self._modelFile)

        print("Training completed")
        print(datetime.datetime.now())

    def gridSearchandCrossValidate(self, allTrainData, retrain=True):
        print("Setting up Y and X for logistic regression")
        print(datetime.datetime.now())
        yTrain, xTrain = patsy.dmatrices(self._regressionFormulaY + ' ~ ' + self._regressionFormulaX, allTrainData,
                                         return_type="dataframe")
        print((xTrain.columns))
        print("No of features in input matrix: %d" % len(xTrain.columns))

        # flatten y into a 1-D array
        print("Flatten y into 1-D array")
        print(datetime.datetime.now())
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
        print("Training model..")
        print(datetime.datetime.now())
        if(retrain):
            self._model = optimized_LR.fit(xTrain, yTrain)
            super(LinearBidModel, self).saveModel(self._model, self._modelFile)
        else:
            self._model = super(LinearBidModel, self).loadSavedModel(self._modelFile)
        print("Training complete")
        print(datetime.datetime.now())

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


# if __name__ == "__main__":
#     # load datasets
#     print("Reading dataset...")
#     trainset = "../dataset/train.csv"
#     validationset = "../dataset/validation.csv"
#     testset = "../dataset/test.csv"
#     # trainDF = ipinyouReader.ipinyouReader(trainset).getDataFrame()
#     reader_encoded = ipinyouReader.ipinyouReaderWithEncoding()
#     trainDF, validateDF, testDF, lookupDict = reader_encoded.getTrainValidationTestDD(trainset, validationset, testset)
#     print("Training dataset...")
#     lrBidModel=LogisticRegressionBidModel()
#     lrBidModel.gridSearchandCrossValidate(trainDF)
#     lrBidModel.trainModel(trainDF)
#     # print("trainDF.info(): ", trainDF.info())








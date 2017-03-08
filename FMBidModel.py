## Factorisation Machine model skeleton for Problem 4.
## version 0.1

"""
## Overview for problem 4 (Use similar structure as problem 3)
1. Create a Factorisation Machine model for CTR estimation
    y ~ x
    where y = click
    where x = a list of variables we choose
    TODO:

2. CTR estimation
    - Estimate click for every record in the test set using the model above
    - Compute pCTR = prob(click=1)

3. Compute bid = base_bid x pCTR/avgCTR
where base_bid assumed avgBudget, avgCTR assumed CTR of training set.

# #List of column names. To be copy and pasted (as needed) in the formula for Factorisation Machine classification
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
from ipinyouWriter import ResultWriter as ResultWriter
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from UserException import ModelNotTrainedException
import datetime
import pandas as pd
from fastFM import sgd
from fastFM import als
import scipy as scipy
from BidModels import BidModelInterface

class FMBidModel(BidModelInterface):
    _regressionFormulaY =''
    _regressionFormulaX =''
    _model=None
    _cBudget=0
    _avgCTR=0
    _modelType=None


    def __init__(self, regressionFormulaY='click', regressionFormulaX='weekday + hour + region + city + adexchange +slotwidth + slotheight + slotprice + advertiser',cBudget=25000*1000, avgCTR=0.000754533880574, modelType="fmclassificationsgd"):
        """

        :param regressionFormulaY:
        :param regressionFormulaX:
        :param cBudget:
        :param avgCTR:
        :param modelType: Options ['fmclassification']
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
        # base_bid_factor=1
        # if (pCTR<0.5):
        #     base_bid_factor=
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
        xTest=testDF[self._regressionFormulaX]

        print("Converting to sparse matrix")
        xTest = scipy.sparse.csc_matrix(xTest.as_matrix())

        # predict click labels for the test set
        print("Predicting test set...")

        # FastFM only give a probabilty of a click=1
        predictedClickOneProb = self._model.predict_proba(xTest)
        # predictedClickOne = self._model.predict(xTest)
        # print("predictedClickOne shape:",predictedClickOne.shape)
        # print("predictedClickOne :", predictedClickOne)
        #
        # print("predictedClickOneProb shape:",predictedClickOneProb.shape)
        # print("predictedClickOneProb :", predictedClickOneProb)
        return predictedClickOneProb


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
        :param modelFile: To save trained model into physical file.
        :return:
        """
        self._modelFile=modelFile
        # instantiate a logistic regression model
        # TODO: Need to tune the model parameters
        if(self._modelType=='fmclassificationals'):
            print("Factorisation Machine with ALS solver will be used for training")
            self._model = als.FMClassification(n_iter=500, rank=2)

        elif(self._modelType=='fmclassificationsgd'):
            print("Factorisation Machine with SGD solver will be used for training")
            self._model = sgd.FMClassification(n_iter=500, rank=2)

        else:
            print("Unrecognised modelType: Factorisation Machine with ALS defaulted training")
            self._model = als.FMClassification(n_iter=500, rank=2)

        if (retrain):
            print("Setting up Y and X for training")
            print(datetime.datetime.now())
            xTrain=allTrainData[self._regressionFormulaX]
            yTrain = allTrainData['click']

            print("Converting X to sparse matrix, required by FastFM")
            xTrain= scipy.sparse.csc_matrix(xTrain.as_matrix())

            # #FastFM can only accept sparse matrixes...so cannot use patsy alone to encode the feature set
            # yTrain, xTrain = patsy.dmatrices(self._regressionFormulaY + ' ~ ' + self._regressionFormulaX, allTrainData, return_type="dataframe")

            # print("No of features in input matrix: %d" % len(xTrain.columns))

            print("Training Model...")
            print(datetime.datetime.now())

            self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear
            super(FMBidModel, self).saveModel(self._model, self._modelFile)

            # check the accuracy on the training set
            print("\n\nTraining acccuracy: %5.3f" % self._model.score(xTrain, yTrain))
        else:
            self._model=super(FMBidModel, self).loadSavedModel(self._modelFile)

        print("Training completed")
        print(datetime.datetime.now())

    def gridSearchandCrossValidate(self, allTrainData, retrain=True):
        #TODO: WARNING: gridSearchandCrossValidate METHOD HAS YET TO BE THOROUGHLY TESTED on this Model
        print("WARNING: gridSearchandCrossValidate METHOD HAS YET TO BE THOROUGHLY TESTED")
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

        param_grid = [{
                          'n_iter': [100],
                          'init_std': [0.095, 0.1, 0.105],
                          'rank':[2],
                          'random_state': [123],
                          'l2_reg_w': [0],
                          'l2_reg_V': [0],
                          'l2_reg': [None],
                          'step_size': [0.1],

                    }
                      ]

        optimized_LR = GridSearchCV(sgd.FMClassification(),
                                     param_grid=param_grid,
                                     scoring='accuracy',
                                     cv=5,
                                     n_jobs=-1,
                                     error_score='raise')
        print("Training model..")
        print(datetime.datetime.now())
        if(retrain):
            self._model = optimized_LR.fit(xTrain, yTrain)
            super(FMBidModel, self).saveModel(self._model, self._modelFile)
        else:
            self._model = super(FMBidModel, self).loadSavedModel(self._modelFile)
        print("Training complete")
        print(datetime.datetime.now())

        scores = optimized_LR.grid_scores_
        # print(type(scores))
        for i in range(len(scores)):
            print(optimized_LR.grid_scores_[i])

    def validateModel(self, allValidateDataOneHot, allValidateData):
        if(self._model!=None):
            print("Setting up X Y validation for prediction")

            xValidate = allValidateDataOneHot[self._regressionFormulaX]

            print("Converting to sparse matrix")
            xValidate = scipy.sparse.csc_matrix(xValidate.as_matrix())

            # predict click labels for the validation set
            print("Predicting validation set...")
            predicted = self._model.predict(xValidate)

            print("Writing to validated prediction csv")
            valPredictionWriter = ResultWriter()
            valPredictionWriter.writeResult(filename="FastFMpredictValidate.csv", data=predicted)
            print("\n\nPrediction report on validation set:",metrics.classification_report(allValidateData['click'], predicted))
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








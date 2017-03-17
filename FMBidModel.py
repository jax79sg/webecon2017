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
from fastFM import mcmc
from fastFM import als
import scipy as scipy
from BidModels import BidModelInterface
import Evaluator
from polylearn import FactorizationMachineClassifier
from ImbalanceLearn import ImbalanceSampling
from Utilities import Utility
import ipinyouReader
import ipinyouWriter
import Evaluator
from  sklearn.metrics import roc_auc_score
from sgdFMClassification import SGDFMClassification

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


    def trainModel(self, X,y, retrain=True, modelFile=None):
        """
        Train model using Logistic Regression for Click against a set of features
        Trained model will be saved to disk (No need retrain/reload training data in future if not required during program rerun)
        :param allTrainData:
        :param retrain: If False, will load self._modelFile instead of training the dataset.
        :param modelFile: To save trained model into physical file.
        :return:
        """
        self._modelFile=modelFile
        print("Getting xTrain")
        xTrain = X
        yTrain = y
        print("xTrain:", xTrain.shape,list(xTrain))
        print("yTrain:", yTrain.shape,set(yTrain['click']),"ListL",list(yTrain))
        yTrain['click'] = yTrain['click'].map({0: -1, 1: 1})


        xTrain.to_csv("data.pruned/xTrain.csv")
        yTrain.to_csv("data.pruned/yTrain.csv")

        print("xTrain:",list(xTrain))
        xTrain=xTrain.as_matrix()
        yTrain = yTrain['click'].as_matrix()
        # print("Performing oversampling to even out")
        # xTrain,yTrain=ImbalanceSampling().oversampling_SMOTE(X=xTrain,y=yTrain)
        xTrain, yTrain = ImbalanceSampling().oversampling_ADASYN(X=xTrain, y=yTrain)

        # instantiate a logistic regression model
        # TODO: Need to tune the model parameters. SGD FastFM still perform better in terms of speed and AUC. Shall stick with it.
        if(self._modelType=='fmclassificationals'):
            print("Factorisation Machine with ALS solver will be used for training")
            print("Converting X to sparse matrix, required by FastFM")
            xTrain= scipy.sparse.csc_matrix(xTrain)
            self._model = als.FMClassification(n_iter=3000, rank=2, verbose=1)

        elif(self._modelType=='fmclassificationsgd'):
            print("Factorisation Machine with SGD solver will be used for training")
            print("Converting X to sparse matrix, required by FastFM")
            xTrain= scipy.sparse.csc_matrix(xTrain)

            # Optimal {'l2_reg_w': 0.0005, 'l2_reg': 0.0005, 'step_size': 0.01, 'n_iter': 200000, 'l2_reg_V': 0.0005}
            # self._model = sgd.FMClassification(n_iter=100000, rank=2, l2_reg_w=0.01, l2_reg_V=0.01, l2_reg=0.01, step_size=0.004)
            self._model = SGDFMClassification(n_iter=200000, rank=2, l2_reg_w=0.0005, l2_reg_V=0.0005, l2_reg=0.0005,
                                               step_size=0.01)

        elif(self._modelType=='polylearn'):
            print("Factorisation Machine from scitkit-learn-contrib polylearn will be used for training")
            self._model = FactorizationMachineClassifier(degree=2, loss='squared_hinge', n_components=2, alpha=1,
                 beta=1, tol=1e-3, fit_lower='explicit', fit_linear=True,
                 warm_start=False, init_lambdas='ones', max_iter=5000,
                 verbose=True, random_state=None)

        else:
            print("Unrecognised modelType: Factorisation Machine with SGD defaulted training")
            print("Converting X to sparse matrix, required by FastFM")
            xTrain= scipy.sparse.csc_matrix(xTrain.as_matrix())
            self._model = SGDFMClassification(n_iter=200000, rank=2, l2_reg_w=0.0005, l2_reg_V=0.0005, l2_reg=0.0005,
                                              step_size=0.01)

        if (retrain):
            print("Setting up Y and X for training")
            print(datetime.datetime.now())

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



    def gridSearchandCrossValidateFastSGD(self, X,y, retrain=True):
        # n_iter=100000, rank=2, l2_reg_w=0.01, l2_reg_V=0.01, l2_reg=0.01, step_size=0.004
        print("Getting xTrain")
        xTrain = X
        yTrain = y
        print("xTrain:", xTrain.shape,list(xTrain))
        print("yTrain:", yTrain.shape,set(yTrain['click']),"ListL",list(yTrain))
        yTrain['click'] = yTrain['click'].map({0: -1, 1: 1})


        xTrain.to_csv("data.pruned/xTrain.csv")
        yTrain.to_csv("data.pruned/yTrain.csv")

        print("xTrain:",list(xTrain))
        xTrain=xTrain.as_matrix()
        yTrain = yTrain['click'].as_matrix()
        print("Performing oversampling to even out")
        # xTrain,yTrain=ImbalanceSampling().oversampling_SMOTE(X=xTrain,y=yTrain)
        xTrain, yTrain = ImbalanceSampling().oversampling_ADASYN(X=xTrain, y=yTrain)

        print("Factorisation Machine with SGD solver will be used for training")
        print("Converting X to sparse matrix, required by FastFM")
        xTrain = scipy.sparse.csc_matrix(xTrain)

        param_grid = [{
                          'n_iter': [5000,10000,15000,20000,25000,50000],
                          'l2_reg_w': [0.0005,0.001,0.005,0.01,0.05,0.1],
                          'l2_reg_V': [0.0005,0.001,0.005,0.01,0.05,0.1],
                          'l2_reg': [0.0005,0.001,0.005,0.01,0.05,0.1],
                          'step_size': [0.0001,0.0005,0.001,0.004,0.01,0.05,0.1]
                        # 'n_iter': [5000],
                        # 'l2_reg_w': [0.0005, 0.001],
                        # 'l2_reg_V': [0.0005, 0.001],
                        # 'l2_reg': [0.0005],
                        # 'step_size': [ 0.004]

        }
                      ]

        optimized_LR = GridSearchCV(SGDFMClassification(),
                                     param_grid=param_grid,
                                     scoring='roc_auc',
                                     cv=5,
                                     # n_jobs=-1,
                                     error_score='raise')
        print("Training model..")
        print(datetime.datetime.now())
        if(retrain):
            self._model = optimized_LR.fit(xTrain, yTrain)
        print("Training complete")
        print(datetime.datetime.now())

        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)

    def validateModel(self, xVal, yVal):
        if(self._model!=None):
            print("Setting up X Y validation for prediction")

            xValidate = xVal
            yVal['click'] = yVal['click'].map({0: -1, 1: 1})

            xVal=xVal.reset_index(drop=True)
            yVal = yVal.reset_index(drop=True)

            click1list=yVal[yVal['click'] == 1].index.tolist()
            click0list = yVal[yVal['click'] == -1].index.tolist()
            print("yVal:", (yVal).shape)
            print("click1list:",len(click1list))
            print("click1list:", len(click0list))

            print("Converting to sparse matrix")
            xValidate = scipy.sparse.csc_matrix(xValidate.as_matrix())

            # predict click labels for the validation set
            print("Predicting validation set...")
            predicted = self._model.predict(xValidate)
            predictedProb = self._model.predict_proba(xValidate)

            predictedOneProbForclick1=predictedProb[click1list][:,1]
            predictedOneProbForclick0 = predictedProb[click0list][:,1]
            print("predictedProbclick1:",(predictedOneProbForclick1).shape)
            print("predictedProbclick0:", (predictedOneProbForclick0).shape)
            print("yVal['click']",yVal['click'].shape)
            print("predictedProb:",predictedProb.shape)
            print("roc_auc",roc_auc_score(yVal['click'], predictedProb[:,1]))

            #Get the Goldclick==1 and retrieve the predictedProb1 for it
            Evaluator.ClickEvaluator().clickProbHistogram(predictedOneProbForclick1,title='Click=1',showGraph=True)

            # Get the Goldclick==0 and retrieve the predictedProb1 for it
            Evaluator.ClickEvaluator().clickProbHistogram(predictedOneProbForclick0,title='Click=0',showGraph=True)

            Evaluator.ClickEvaluator().clickROC(yVal['click'],predictedProb[:,1],showGraph=True)

            print("Gold label: ",yVal['click'])
            print("predicted label: ", predicted)

            print("Writing to validated prediction csv")
            valPredictionWriter = ResultWriter()
            valPredictionWriter.writeResult(filename="data.pruned/FastFMpredictValidate.csv", data=predicted)
            print("yVal['click']:",yVal['click'].shape,yVal['click'].head(2))
            print("\n\nPrediction report on validation set:",metrics.classification_report(yVal['click'], predicted))
            # Evaluator.ClickEvaluator().printClickPredictionScore(y_Gold=allValidateDataOneHot['click'],y_Pred=predicted)

        else:
            print("Error: No model was trained in this instance....")


if __name__ == "__main__":
    trainset = "data.final/train1_cleaned_prune.csv"
    validationset = "data.final/validation_cleaned.csv"
    testset = "../dataset/empty.csv"

    print("Reading dataset...")
    timer = Utility()
    timer.startTimeTrack()

    trainReader = ipinyouReader.ipinyouReader(trainset)
    validationReader = ipinyouReader.ipinyouReader(validationset)

    trainOneHotData, trainY = trainReader.getOneHotData()
    validationOneHotData, valY = validationReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist())
    timer.checkpointTimeTrack()

    print("trainOneHotData:",trainOneHotData.shape,list(trainOneHotData))


    # for colname in list(trainOneHotData):
    #     if "creative_" in colname:
    #         trainOneHotData=trainOneHotData.drop([colname])
    #
    #     if "ip_block_" in colname:
    #         trainOneHotData=trainOneHotData.drop([colname])
    #
    #     if "keypage_" in colname:
    #         trainOneHotData=trainOneHotData.drop([colname])
    #
    #     if "null" in colname:
    #         trainOneHotData=trainOneHotData.drop([colname])
    #
    # for colname in list(validationOneHotData):
    #     if "creative_" in colname:
    #         validationOneHotData=validationOneHotData.drop([colname])
    #
    #     if "ip_block_" in colname:
    #         validationOneHotData=validationOneHotData.drop([colname])
    #
    #     if "keypage_" in colname:
    #         validationOneHotData=validationOneHotData.drop([colname])
    #
    #     if "null" in colname:
    #         validationOneHotData=validationOneHotData.drop([colname])


    print("trainY:", trainY.shape, list(trainY))
    print("validationOneHotData:",validationOneHotData.shape,list(validationOneHotData))
    print("valY:", valY.shape, list(valY))

    fmBidModel=FMBidModel(regressionFormulaY='click', regressionFormulaX=list(trainOneHotData), cBudget=272.412385 * 1000, avgCTR=0.2, modelType='fmclassificationsgd')
    # fmBidModel.gridSearchandCrossValidateFastSGD(trainOneHotData, trainY)

    timer.startTimeTrack()
    fmBidModel.trainModel(trainOneHotData,trainY, retrain=True, modelFile="data.pruned/fmclassificationsgd.pkl")
    timer.checkpointTimeTrack()
    fmBidModel.validateModel(validationOneHotData, valY)
    timer.checkpointTimeTrack()







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
from ipinyouWriter import ResultWriter as ResultWriter
from sklearn.grid_search import GridSearchCV
from UserException import ModelNotTrainedException
import datetime
import pandas as pd
from fastFM import als
import scipy as scipy
from BidModels import BidModelInterface
from polylearn import FactorizationMachineClassifier
from ImbalanceLearn import ImbalanceSampling
from Utilities import Utility
import ipinyouReader
import Evaluator
from  sklearn.metrics import roc_auc_score
from sgdFMClassification import SGDFMClassification
from BidPriceEstimator import BidEstimator
from sklearn.metrics import confusion_matrix


class FMBidModel(BidModelInterface):
    # _regressionFormulaY =''
    # _regressionFormulaX =''
    _model=None
    _cBudget=0
    _modelType=None


    def __init__(self,cBudget=6250*1000, modelType="fmclassificationsgd"):
        """

        # :param regressionFormulaY:
        # :param regressionFormulaX:
        :param cBudget:
        # :param avgCTR:
        :param modelType: Options ['fmclassificationsgd','fmclassificationals','polylearn']
        """
        # self._regressionFormulaY=regressionFormulaY
        # self._regressionFormulaX = regressionFormulaX
        # self._defaultBid = 0
        self._cBudget=cBudget
        # self._avgCTR=avgCTR
        self._modelType=modelType

    def getThreshold(self):
        return 0.5

    def __computeBidPrice(self, pCTR=None):
        """
        The default computation to compute bid price
        The implemented model should have its own ways to gather the necessary parameters as follows
        :param basebid:Using the budget in this case
        :param pCTR: Compute the probability that click=1 for that bidrequest
        :param avgCTR: Consider this as the avgCTR for the training set
        :return: bid
        """
        bid=BidEstimator().linearBidPrice_mConfi(y_pred=pCTR, base_bid=self._cBudget, m_conf=0.8,variable_bid=10)
        print("Bid type:",type(bid))
        return bid

    def __predictClickOneProb(self,testDF):
        """
        Perform prediction for click label.
        Take the output of click=1 probability as the CTR.
        :param oneBidRequest:
        :return:
        """

        print("Setting up X test for prediction")
        xTest=testDF

        print("Converting to sparse matrix")
        xTest = scipy.sparse.csc_matrix(xTest.as_matrix())

        # predict click labels for the test set
        print("Predicting test set...")

        # FastFM only give a probabilty of a click=1
        predictedClickOneProb = self._model.predict_proba(xTest)

        return predictedClickOneProb

    def __predictClickOne(self,testDF):
        """
        Perform prediction for click label.
        Take the output of click=0 or 1 as the CTR.
        :param oneBidRequest:
        :return:
        """

        print("Setting up X test for prediction")
        xTest=testDF

        print("Converting to sparse matrix")
        xTest = scipy.sparse.csc_matrix(xTest.as_matrix())

        # predict click labels for the test set
        print("Predicting test set...")

        # FastFM only give a probabilty of a click=1
        predictedClick = self._model.predict(xTest, self.getThreshold())

        return predictedClick

    def trimToBudget(self, bidpriceDF, budget):
        """
        In case the bidding process exceeds the budget, trim down the bidding
        :param bidpriceDF:
        :param budget:
        :return:
        """
        print("Trimming....")
        totalspend=np.sum(bidpriceDF)
        overspend = totalspend - budget
        print("bidpriceDF:",bidpriceDF.shape)
        print("budget:",budget)
        print("totalspend:", totalspend)
        print("overspend:", overspend)
        i = -1
        while overspend > 0 and len(bidpriceDF) + i > 0:
            overspend += -bidpriceDF[i]
            bidpriceDF[i] = 0
            i += -1

        print("bidpriceDF:",bidpriceDF)
        print("np.sum(bidpriceDF:",np.sum(bidpriceDF))
        assert(np.sum(bidpriceDF)<budget)
        return bidpriceDF


    def getBidPrice(self, xTestOneHotDF, yValDF,noBidThreshold=0.2833333,minBid=200,bidRange=90,sigmoidDegree=-10):
        """
        Retrieve the bidding price
        :param xTestOneHotDF:
        :param yValDF:
        :param noBidThreshold:
        :param minBid:
        :param bidRange:
        :param sigmoidDegree:
        :return:
        """
        print("Computing bid price")
        print("xTestOneHotDF:",xTestOneHotDF.shape,list(xTestOneHotDF))
        print("yValDF:", yValDF.shape, list(yValDF))
        if(self._model==None):
            raise ModelNotTrainedException("Model must be trained prior to prediction!")

        pCTR = self.__predictClickOneProb(xTestOneHotDF)[:, 1] #Prob of click==1
        bidprice = BidEstimator().thresholdSigmoid(predOneProb=pCTR,noBidThreshold=0.2833333,minBid=200,bidRange=90,sigmoidDegree=-10)
        print("bidprice:",bidprice)
        bidprice = self.trimToBudget(bidprice,self._cBudget)
        print("bidprice after trim:", bidprice)

        #merge with bidid
        bidpriceDF=pd.DataFrame(bidprice,columns=['bidprice'])
        print("bidpriceDF:",bidpriceDF.shape,list(bidpriceDF))
        bididDF=pd.DataFrame(yValDF['bidid'],columns=['bidid'])
        print("bididDF:", bididDF.shape, list(bididDF))
        bidIdPriceDF=pd.concat([bididDF,bidpriceDF],axis=1,ignore_index=True)
        print("bidIdPriceDF:",bidIdPriceDF.shape,list(bidIdPriceDF))
        return bidIdPriceDF

    # def getBidPrice(self, allBidRequest):
    #     """
    #     1. Predict click=1 prob for entire test/validation set
    #         Considered as pCTR for each impression
    #     2. Use the bid=base_price*(pCTR/avgCTR) formula
    #     :param oneBidRequest:
    #     :return:
    #     """
    #
    #     if(self._model==None):
    #         raise ModelNotTrainedException("Model must be trained prior to prediction!")
    #
    #
    #
    #     #Compute the CTR of this BidRequest
    #     pCTR=self.__predictClickOneProb(allBidRequest)[:,1]
    #     print("General sensing of pCTR ranges")
    #     print(pCTR)
    #
    #     #Compute the bid price
    #     bids = np.apply_along_axis(self.__computeBidPrice, axis=0, arr=pCTR)
    #     print("General sensing of bids ranges")
    #     print(bids)
    #
    #     #Extract the corresponding bidid
    #     allBidRequestMatrix=allBidRequest.as_matrix(columns=['bidid'])
    #
    #     #Merging bidid and bids into a table (Needed for eval)
    #     bidid_bids=np.column_stack((allBidRequestMatrix, bids))
    #
    #     bids = pd.DataFrame(bidid_bids, columns=['bidid', 'bidprice'])
    #     return bids


    def trainModel(self, X,y, retrain=True, modelFile=None):
        """
        Train model using FM for Click against a set of features
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

        if(retrain):
            print("Performing oversampling to even out")
            xTrain,yTrain=ImbalanceSampling().oversampling_SMOTE(X=xTrain,y=yTrain)
            #ADASYN is slower and doesn't offer better model performance, choose SMOTE instead.
            # xTrain, yTrain = ImbalanceSampling().oversampling_ADASYN(X=xTrain, y=yTrain)

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
            print("Training with n_iter=200000, rank=2, l2_reg_w=0.0005, l2_reg_V=0.0005, l2_reg=0.0005,step_size=0.01")

            # Best Training set score: 0.9121148444887212
            # Best Param: {'n_iter': 200000, 'l2_reg_w': 0.0005, 'step_size': 0.004, 'l2_reg_V': 0.005, 'rank': 16}
            self._model = SGDFMClassification(n_iter=200000, rank=16, l2_reg_w=0.0005, l2_reg_V=0.0005, l2_reg=0.0005,step_size=0.01)

        elif(self._modelType=='polylearn'):
            print("Factorisation Machine from scitkit-learn-contrib polylearn will be used for training")
            self._model = FactorizationMachineClassifier(degree=2, loss='squared_hinge', n_components=2, alpha=1,
                 beta=1, tol=1e-3, fit_lower='explicit', fit_linear=True,
                 warm_start=False, init_lambdas='ones', max_iter=5000,
                 verbose=True, random_state=None)

        else:
            raise ModelNotTrainedException('Selected model not available','Valid models are polylearn,fmclassificationsgd,fmclassificationals')

        if (retrain):
            print("Setting up Y and X for training")
            print(datetime.datetime.now())

            print("Training Model...")
            print(datetime.datetime.now())

            self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear
            super(FMBidModel, self).saveModel(self._model, self._modelFile)

        else:
            self._model=super(FMBidModel, self).loadSavedModel(self._modelFile)

        print("Training completed")
        print(datetime.datetime.now())

    def optimiseBid(self, xTestDF,yTestDF):
        """
        Perform bid optimisation based on params
        :param xTestDF:
        :param yTestDF:
        :return:
        """
        print(" xTestDF:",xTestDF.shape,"\n",list(xTestDF))
        print(" yTestDF:", yTestDF.shape, "\n", list(yTestDF))
        result = pd.concat([xTestDF, yTestDF], axis=1)
        print(" result:", result.shape, "\n", list(result))
        predProb = self.__predictClickOneProb(xTestDF)
        be = BidEstimator()
        be.gridSearch_bidPrice(predProb[:,1], 0, 0, result, bidpriceest_model='thresholdsigmoid')


    def gridSearchandCrossValidateFastSGD(self, X,y, retrain=True):
        """
        Perform gridsearch on FM model
        :param X:
        :param y:
        :param retrain:
        :return:
        """
        # n_iter=100000, rank=2, l2_reg_w=0.01, l2_reg_V=0.01, l2_reg=0.01, step_size=0.004
        print("Getting xTrain")
        xTrain = X
        yTrain = y
        print("xTrain:", xTrain.shape,list(xTrain))
        print("yTrain:", yTrain.shape,set(yTrain['click']),"ListL",list(yTrain))
        yTrain['click'] = yTrain['click'].map({0: -1, 1: 1})


        # xTrain.to_csv("data.pruned/xTrain.csv")
        # yTrain.to_csv("data.pruned/yTrain.csv")

        print("xTrain:",list(xTrain))
        xTrain=xTrain.as_matrix()
        yTrain = yTrain['click'].as_matrix()
        print("Performing oversampling to even out")
        xTrain,yTrain=ImbalanceSampling().oversampling_SMOTE(X=xTrain,y=yTrain)

        print("Factorisation Machine with SGD solver will be used for training")
        print("Converting X to sparse matrix, required by FastFM")
        xTrain = scipy.sparse.csc_matrix(xTrain)

        param_grid = [{
                          'n_iter': [150000,200000,250000],
                          'l2_reg_w': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
                          'l2_reg_V': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
                          # 'l2_reg': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
                          'step_size': [0.0005,0.004,0.007],
                          'rank':[32,36,42,46,52,56,64]
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
                                     error_score='raise',
                                    verbose=1)
        print("Training model..")
        print(datetime.datetime.now())
        if(retrain):
            self._model = optimized_LR.fit(xTrain, yTrain)
        print("Training complete")
        print(datetime.datetime.now())

        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)

    def validateModel(self, xVal, yVal):
        """
        Perform validation of model with different metrics and graphs for analysis
        :param xVal:
        :param yVal:
        :return:
        """
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

            #Convert -1 to 0 as Evaluator printClickPredictionScore cannot handle -1
            predicted[predicted==-1] = 0
            yVal['click'] = yVal['click'].map({-1: 0, 1: 1})
            Evaluator.ClickEvaluator().printClickPredictionScore(predicted,yVal['click'])

            cnf_matrix = confusion_matrix(yVal['click'], predicted)

            Evaluator.ClickEvaluator().plot_confusion_matrix(cm=cnf_matrix,classes=set(yVal['click']),plotgraph=True,printStats=True)
            #Change back, just in case
            predicted[predicted==0] = -1
            yVal['click'] = yVal['click'].map({0: -1, 1: 1})

            print("Gold label: ",yVal['click'])
            print("predicted label: ", predicted)

            print("Writing to validated prediction csv")
            valPredictionWriter = ResultWriter()
            valPredictionWriter.writeResult(filename="data.pruned/FastFMpredictValidate.csv", data=predicted)

        else:
            print("Error: No model was trained in this instance....")


if __name__ == "__main__":

    trainset = "data.final/train1_cleaned_prune.csv"
    validationset = "data.final/validation_cleaned.csv"
    testset = "data.final/test.csv"

    print("Reading dataset...")
    timer = Utility()
    timer.startTimeTrack()

    trainReader = ipinyouReader.ipinyouReader(trainset)
    validationReader = ipinyouReader.ipinyouReader(validationset)
    testReader = ipinyouReader.ipinyouReader(testset)
    timer.checkpointTimeTrack()
    print("Getting encoded datasets")
    trainOneHotData, trainY = trainReader.getOneHotData()
    validationOneHotData, valY = validationReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist())
    testOneHotData, testY = testReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist())
    timer.checkpointTimeTrack()

    print("trainOneHotData:",trainOneHotData.shape,list(trainOneHotData))



    print("trainY:", trainY.shape, list(trainY))
    print("validationOneHotData:",validationOneHotData.shape,list(validationOneHotData))
    print("valY:", valY.shape, list(valY))

    fmBidModel=FMBidModel(cBudget=6250 * 1000, modelType='fmclassificationsgd')
    print("==========Training starts")
    # fmBidModel.gridSearchandCrossValidateFastSGD(trainOneHotData, trainY)


    fmBidModel.trainModel(trainOneHotData,trainY, retrain=True, modelFile="data.pruned/fmclassificationsgd.pkl")
    timer.checkpointTimeTrack()

    # print("==========Validation starts")
    # fmBidModel.validateModel(validationOneHotData, valY)
    # timer.checkpointTimeTrack()


    print("==========Bid optimisation starts")
    fmBidModel.optimiseBid(validationOneHotData,valY)
    timer.checkpointTimeTrack()

    # best score      0.3683528286042599
    # noBidThreshold  2.833333e-01
    # minBid          2.000000e+02
    # bidRange        9.000000e+01
    # sigmoidDegree - 1.000000e+01
    # won             3.432900e+04
    # click           1.380000e+02
    # spend           2.729869e+06
    # trimmed_bids    0.000000e+00
    # CTR             4.019925e-03
    # CPM             7.952078e+04
    # CPC             1.978166e+04
    # blended_score   3.683528e-01

    # best score      0.3681133881545131
    # noBidThreshold  2.833333e-01
    # minBid          2.000000e+02
    # bidRange        1.000000e+02
    # sigmoidDegree - 1.000000e+01
    # won             3.449900e+04
    # click           1.380000e+02
    # spend           2.758561e+06
    # trimmed_bids    0.000000e+00
    # CTR             4.000116e-03
    # CPM             7.996061e+04
    # CPC             1.998957e+04
    # blended_score   3.681134e-01


    # New budget      6250000
    # FM
    # best score      0.32755084132163526
    # noBidThreshold  8.666667e-01
    # minBid          2.000000e+02
    # bidRange        2.500000e+02
    # sigmoidDegree - 1.000000e+01
    # won             1.461000e+04
    # click           1.170000e+02
    # spend           1.124960e+06
    # trimmed_bids    0.000000e+00
    # CTR             8.008214e-03
    # CPM             7.699932e+04
    # CPC             9.615043e+03
    # blended_score   3.275508e-01

    # print("==========Getting  bids")
    # # bidIdPriceDF=fmBidModel.getBidPrice(validationOneHotData,valY,noBidThreshold=0.2833333,minBid=200,bidRange=100,sigmoidDegree=-10)
    # bidIdPriceDF=fmBidModel.getBidPrice(validationOneHotData,valY,noBidThreshold=0.8666667,minBid=200,bidRange=250,sigmoidDegree=-10)
    # print("bidIdPriceDF:",bidIdPriceDF.shape, list(bidIdPriceDF))
    # bidIdPriceDF.to_csv("mybids.csv")
    # timer.checkpointTimeTrack()






import ipinyouReader
from sklearn.grid_search import GridSearchCV
import Evaluator
from sklearn.linear_model import SGDClassifier
from Evaluator import ClickEvaluator
import numpy as np
from UserException import ModelNotTrainedException
from BidModels import BidModelInterface
from BidPriceEstimator import BidEstimator
import pandas as pd

class LinearBidModel_v2(BidModelInterface):
    _model=None

    def __init__(self, cBudget, avgCTR):
        self._cBudget = cBudget
        self._avgCTR = avgCTR

    def getBidPrice(self, allBidRequest, v_df):
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
        y_pred = self._model.predict_proba(allBidRequest)
        y_pred = y_pred[:, 1]

        bidprice = BidEstimator().linearBidPrice(y_pred, self._cBudget, self._avgCTR)

        bids = np.stack([v_df['bidid'], bidprice], axis=1)
        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])
        print(bids.info())
        return y_pred, bids

    # def getY_Pred(self, allBidRequest):
    #     if(self._model==None):
    #         raise ModelNotTrainedException("Model must be trained prior to prediction!")
    #
    #     #Compute the CTR of this BidRequest
    #     pred = self._model.predict_proba(allBidRequest)
    #     pred = pred[:, 1]
    #
    #     return pred

    def trainModel(self, xTrain, yTrain):
        self._model = SGDClassifier(alpha=0.0005, penalty='l2', loss='log', n_iter=200)
        # self._model = SGDClassifier(alpha=0.0015, penalty='l1', loss='log', n_iter=100)
        self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear

        pred = self._model.predict_proba(xTrain)
        pred = pred[:, 1]

        ce = Evaluator.ClickEvaluator()
        ce.printRMSE(pred, yTrain)
        ce.clickROC(yTrain, pred, False)
        pred = [1 if i >= 0.5 else 0 for i in pred]
        ce.printClickPredictionScore(pred, yTrain)

    def validateModel(self, xValidate, yValidate, validateDF):

        pred = self._model.predict_proba(xValidate)
        pred = pred[:, 1]

        ce = Evaluator.ClickEvaluator()
        ce.printRMSE(pred, yValidate)
        ce.clickROC(yValidate, pred, False)
        click1 = pred[validateDF.click == 1]
        n, bins, patches = ClickEvaluator().clickProbHistogram(pred_prob=click1, color='g',
                                                         title='Predicted probabilities for clicks=1',
                                                         # imgpath="./SavedCNNModels/xgboost-click1-" + bidmodel.timestr + ".jpg",
                                                         showGraph=True)

        # click=0 prediction as click=1 probabilities
        click0 = pred[validateDF.click == 0]
        n, bins, patches = ClickEvaluator().clickProbHistogram(pred_prob=click0, color='r',
                                                         title='Predicted probabilities for clicks=0',
                                                         # imgpath="./SavedCNNModels/xgboost-click0-" + bidmodel.timestr + ".jpg",
                                                         showGraph=True)
        pred = [1 if i >= 0.5 else 0 for i in pred]
        ce.printClickPredictionScore(pred, yValidate)



    def gridSearchandCrossValidate(self, xTrain, yTrain):
        ## Setup Grid Search parameter
        # param_grid = [{
        #                   'solver': ['liblinear'],
        #                   'C': [0.15, 0.19, 0.2, 0.21, 0.25],
        #                   'class_weight':[None],  # None is better
        #                   'penalty': ['l2', 'l1'],
        #               }

        param_grid = [{
                            'alpha':[0.0050, 0.0015, 0.0025],
                            'penalty': ['l1', 'l2', 'elasticnet'],
                            'n_iter': [50, 100, 200]
        }

                      #   ,
                      # {
                      #     'solver': ['newton-cg', 'lbfgs', 'sag'],
                      #     'C': [0.1, 0.5, 1.0, 2.0],
                      #     'max_iter':[50000],
                      #     'class_weight': [None],  # None is better
                      #     'penalty': ['l2'],
                      # }
                      ]

        optimized_LR = GridSearchCV(SGDClassifier(),
                                     param_grid=param_grid,
                                     scoring='roc_auc',
                                     cv=3,
                                     n_jobs=-1,
                                     error_score='raise',
                                     verbose=0
                                     )
        print("Grid Searching...")
        self._model = optimized_LR.fit(xTrain, yTrain)
        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)

        scores = optimized_LR.grid_scores_
        for i in range(len(scores)):
            print(optimized_LR.grid_scores_[i])

    def optimiseBid(self, xTestDF,yTestDF):
        print(" xTestDF:",xTestDF.shape,"\n",list(xTestDF))
        print(" yTestDF:", yTestDF.shape, "\n", list(yTestDF))
        result = pd.concat([xTestDF, yTestDF], axis=1)
        print(" result:", result.shape, "\n", list(result))
        predProb = self._model.predict_proba(xTestDF)

        be = BidEstimator()
        be.gridSearch_bidPrice(predProb[:,1], 0.2, 0, result, bidpriceest_model='linearBidPrice')

if __name__ == "__main__":
    trainset="data.final/train1_cleaned_prune.csv"
    validationset="data.final/validation_cleaned.csv"
    testset="data.final/test.csv"
    # trainset="../dataset/debug.csv"
    # validationset="../dataset/debug.csv"
    # testset="../dataset/debug.csv"

    trainReader = ipinyouReader.ipinyouReader(trainset)
    validationReader = ipinyouReader.ipinyouReader(validationset)

    trainOneHotData, trainY = trainReader.getOneHotData()
    validationOneHotData, valY = validationReader.getOneHotData(
        train_cols=trainOneHotData.columns.get_values().tolist())

    X_train = trainOneHotData
    Y_train = trainY['click']
    X_val = validationOneHotData
    Y_val = valY['click']

    lbm = LinearBidModel_v2(cBudget=110, avgCTR=0.2)
    lbm.trainModel(X_train, Y_train)
    # lbm.gridSearchandCrossValidate(X_train, Y_train)
    lbm.validateModel(X_val, Y_val, valY)

    print("==========Bid optimisation starts")
    lbm.optimiseBid(validationOneHotData, valY)
    # lbm.optimiseBid(trainOneHotData, trainY)

    # lbm.getBidPrice(X_val)






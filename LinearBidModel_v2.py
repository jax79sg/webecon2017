import ipinyouReader
from sklearn.grid_search import GridSearchCV
import Evaluator
from sklearn.linear_model import LogisticRegression
import numpy as np
from UserException import ModelNotTrainedException
from BidModels import BidModelInterface
from BidPriceEstimator import BidEstimator
import pandas as pd

class LinearBidModel_v2(BidModelInterface):
    def __init__(self, cBudget, avgCTR):
        self._cBudget = cBudget
        self._avgCTR = avgCTR

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
        pred = self._model.predict_proba(allBidRequest)
        pred = pred[:, 1]

        bidprice = BidEstimator().linearBidPrice(pred, self._cBudget, self._avgCTR)

        bids = np.stack([allBidRequest['bidid'], bidprice], axis=1)
        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

        return bids

    def trainModel(self, xTrain, yTrain):
        self._model = LogisticRegression(C=0.21, penalty='l1', solver='liblinear')
        self._model = self._model.fit(xTrain, yTrain)  # Loss function:liblinear

        pred = self._model.predict_proba(xTrain)
        pred = pred[:, 1]

        ce = Evaluator.ClickEvaluator()
        ce.printRMSE(pred, yTrain)
        ce.roc_results_plot(yTrain, pred, False)
        pred = [1 if i >= 0.5 else 0 for i in pred]
        ce.printClickPredictionScore(pred, yTrain)

    def validateModel(self, xValidate, yValidate):

        pred = self._model.predict_proba(xValidate)
        pred = pred[:, 1]

        ce = Evaluator.ClickEvaluator()
        ce.printRMSE(pred, yValidate)
        ce.roc_results_plot(yValidate, pred, False)
        pred = [1 if i >= 0.5 else 0 for i in pred]
        ce.printClickPredictionScore(pred, yValidate)

    def gridSearchandCrossValidate(self, xTrain, yTrain):
        ## Setup Grid Search parameter
        param_grid = [{
                          'solver': ['liblinear'],
                          'C': [0.15, 0.19, 0.2, 0.21, 0.25],
                          'class_weight':[None],  # None is better
                          'penalty': ['l2', 'l1'],
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

        optimized_LR = GridSearchCV(LogisticRegression(),
                                     param_grid=param_grid,
                                     scoring='accuracy',
                                     cv=5,
                                     n_jobs=-1,
                                     error_score='raise')
        print("Grid Searching...")
        self._model = optimized_LR.fit(xTrain, yTrain)
        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)

        scores = optimized_LR.grid_scores_
        for i in range(len(scores)):
            print(optimized_LR.grid_scores_[i])

if __name__ == "__main__":
    trainset="../dataset/train_cleaned_prune.csv"
    validationset="../dataset/validation_cleaned_prune.csv"
    testset="../dataset/test.csv"
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

    lbm = LinearBidModel_v2(cBudget=272.412385 * 1000, avgCTR=0.2)
    lbm.trainModel(X_train, Y_train)
    # lbm.gridSearchandCrossValidate(X_train, Y_train)
    lbm.validateModel(X_val, Y_val)
    # lbm.getBidPrice(X_val)




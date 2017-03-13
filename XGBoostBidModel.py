
import xgboost as xgb
import numpy as np
import pandas as pd
from ipinyouReader import ipinyouReaderWithEncoding
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import seaborn as sns
from Evaluator import ClickEvaluator
import matplotlib.pyplot as plt
from BidModels import BidModelInterface
from BidPriceEstimator import BidEstimator
import Evaluator

class XGBoostBidModel(BidModelInterface):

    def __init__(self, X_column, Y_column):
        self.Y_column = Y_column
        self.X_column = X_column

    def getBidPrice(self, testDF):
        print("Setting up XGBoost for Test set")
        xTest = testDF[self.X_column]

        xgdmat = xgb.DMatrix(xTest)
        y_pred = self._model.predict(xgdmat)

        # y_pred = [1 if i >= 0.07 else 0 for i in y_pred]

        # bidprice = BidEstimator().linearBidPrice(y_pred, base_bid=220, avg_ctr=0.2)
        bidprice = BidEstimator().linearBidPrice_mConfi(y_pred, base_bid=240, variable_bid=100, m_conf=0.8)

        bids = np.stack([testDF['bidid'], bidprice], axis=1)
        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

        return bids



    def trainModel(self, trainDF):
        print("Setting up XGBoost for Training: X and Y")
        xTrain = trainDF[self.X_column]
        yTrain = trainDF[self.Y_column]

        # print(xTrain.columns)
        print ("No of features in input matrix: %d" % len(xTrain.columns))

        optimised_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                     'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':3, 'learning_rate': 0.045}
        xgdmat = xgb.DMatrix(xTrain, yTrain) # Create our DMatrix to make XGBoost more efficient
        self._model = xgb.train(optimised_params, xgdmat, num_boost_round=432,  verbose_eval=False)

        print("Importance: ", self._model.get_fscore())
        xgdmat = xgb.DMatrix(xTrain)
        y_pred = self._model.predict(xgdmat)

        ClickEvaluator().clickROC(yTrain, y_pred, False)
        ClickEvaluator().printRMSE(y_pred, yTrain)
        y_pred = [1 if i > 0.12 else 0 for i in y_pred]
        ClickEvaluator().printClickPredictionScore(y_pred, yTrain)


        sns.set(font_scale = 1.5)
        # xgb.plot_importance(self._model)

        fscore = self._model.get_fscore()
        importance_frame = pd.DataFrame({'Importance': list(fscore.values()), 'Feature': list(fscore.keys())})
        importance_frame.sort_values(by='Importance', inplace=True)
        importance_frame.plot(kind='barh', x='Feature', figsize=(8, 8), color='orange')
        # plt.show()

    def gridSearch(self, trainDF):
        print("Setting up XGBoost for GridSearch: X and Y")
        xTrain = trainDF[self.X_column]
        yTrain = trainDF[self.Y_column]

        print(xTrain.columns)
        print("No of features in input matrix: %d" % len(xTrain.columns))

        ## Setup Grid Search parameter
        param_grid = {'max_depth': [3, 4],
                      'min_child_weight': [2, 3],
                      'subsample': [0.7, 0.8, 0.9],
                      'learning_rate': [0.04, 0.045]
                      }

        ind_params = {'n_estimators': 1000,
                      'seed': 0,
                      'colsample_bytree': 0.8,
                      'objective': 'binary:logistic',
                      'base_score': 0.5,
                      'colsample_bylevel': 1,
                      'gamma': 0,
                      'max_delta_step': 0,
                      'missing': None,
                      'reg_alpha': 0,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'silent': True,
                      }

        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     param_grid=param_grid,
                                     scoring='accuracy',
                                     cv=5,
                                     n_jobs=-1,
                                     error_score='raise')

        model = optimized_GBM.fit(xTrain, yTrain)
        # check the accuracy on the training set
        print("\n\nTraining acccuracy: %5.3f" % model.score(xTrain, yTrain))
        y_pred = model.predict(xTrain)
        p, r, f1, _ = metrics.precision_recall_fscore_support(yTrain, y_pred)
        # print(p)
        print("Number of 1: ", np.count_nonzero(y_pred))
        print("Number of 0: ", len(y_pred) - np.count_nonzero(y_pred))
        for i in range(len(p)):
            print("Precision: %5.3f \tRecall: %5.3f \tF1: %5.3f" % (p[i], r[i], f1[i]))
        scores = optimized_GBM.grid_scores_
        # print(type(scores))
        for i in range(len(scores)):
            print(optimized_GBM.grid_scores_[i])

    def __estimateClick(self, df):
        xTest = df[self.X_column]
        print("No of features in input matrix: %d" % len(xTest.columns))

        xgdmat = xgb.DMatrix(xTest)
        y_pred = self._model.predict(xgdmat)

        return y_pred

    def validateModel(self, validateDF):
        print("Setting up XGBoost for Validation: X and Y")
        # xValidate = validateDF[self.X_column]
        yValidate = validateDF[self.Y_column]

        # print(xValidate.columns)
        # print("No of features in input matrix: %d" % len(xValidate.columns))

        # optimised_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
        #              'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}
        # xgdmat = xgb.DMatrix(xValidate, yTrain) # Create our DMatrix to make XGBoost more efficient
        # self._model = xgb.train(optimised_params, xgdmat, num_boost_round=432,  verbose_eval=False)
        #
        # print("Importance: ", self._model.get_fscore())

        # xgdmat = xgb.DMatrix(xValidate)
        # y_pred = self._model.predict(xgdmat)

        y_pred = self.__estimateClick(validateDF)

        ClickEvaluator().roc_results_plot(yValidate, y_pred, False)
        ClickEvaluator().printRMSE(y_pred, yValidate)
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        ClickEvaluator().printClickPredictionScore(y_pred, yValidate)

        # sns.set(font_scale = 1.5)
        # xgb.plot_importance(final_gb)

        # fscore = final_gb.get_fscore()
        # importance_frame = pd.DataFrame({'Importance': list(fscore.values()), 'Feature': list(fscore.keys())})
        # importance_frame.sort_values(by='Importance', inplace=True)
        # importance_frame.plot(kind='barh', x='Feature', figsize=(8, 8), color='orange')



    def tunelinearBaseBid(self, testDF):
        print("Setting up XGBoost for Test set")
        y_pred = self.__estimateClick(testDF)

        # y_pred = [1 if i >= 0.07 else 0 for i in y_pred]

        # avgCTR = np.count_nonzero(testDF.click) / testDF.shape[0]
        myEvaluator = Evaluator.Evaluator()

        bestCTR = -1
        bestBidPrice = -1

        print("y_pred mean: ", np.mean(y_pred))

        x = np.arange(0.5, 0.9, 0.05)
        # for i in x:
        for i in range(220, 260):
            print("================= i : ", i)
            # bidprice = BidEstimator().linearBidPrice(y_pred, i, 0.2)
            bidprice = BidEstimator().linearBidPrice_mConfi(y_pred, i, 90, 0.8)
            bids = np.stack([testDF['bidid'], bidprice], axis=1)

            bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

            # print("Estimated bid price: ", bids.bidprice.ix[0])

            resultDict = myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validateDF)
            myEvaluator.printResult()
            ctr = resultDict['click'] / resultDict['won']

            if ctr > bestCTR:
                bestCTR = ctr
                bestBidPrice = i

        print("Best CTR: %.5f \nPrice: %d" %(bestCTR, bestBidPrice))


    def tuneConfidenceBaseBid(self, testDF):
        print("Setting up XGBoost for Test set")
        y_pred = self.__estimateClick(testDF)

        y_pred = [1 if i >= 0.7 else 0 for i in y_pred]

        # print("number of 1 here: ", sum(y_pred))
        # avgCTR = np.count_nonzero(testDF.click) / testDF.shape[0]
        myEvaluator = Evaluator.Evaluator()

        bestCTR = -1
        bestBidPrice = -1
        for i in range(300, 301):
            bidprice = BidEstimator().confidenceBidPrice(y_pred, -1, i)

            # print("total bid price: ", sum(bidprice))
            # print("total bid submitted: ", np.count_nonzero(bidprice))
            # print("Number of $0 bid", bidprice.count(0))

            bids = np.stack([testDF['bidid'], bidprice], axis=1)

            bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

            # print("Estimated bid price: ", bids.bidprice.ix[0])

            resultDict = myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validateDF)
            myEvaluator.printResult()
            ctr = resultDict['click'] / resultDict['won']

            if ctr > bestCTR:
                bestCTR = ctr
                bestBidPrice = i

        print("Best CTR: %.5f \nPrice: %d" % (bestCTR, bestBidPrice))


if __name__ == "__main__":

    # y_pred = [0.4, 0.55, 0.3, 0.51]
    # y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    #
    # print(y_pred)

    # a = np.array([1,0,1,1,1])
    # b = np.array([1,0,1,0,1])
    # c = np.full(b.shape, 1, dtype=int)
    #
    # d=b[np.logical_and(a == 1, b == 1)]
    # d = np.count_nonzero(np.logical_and(a == 1, b == 1))
    #
    # print(d)





    trainset="../dataset/train_cleaned_prune.csv"
    validationset="../dataset/validation_cleaned.csv"
    # testset="../dataset/test.csv"
    # trainset="../dataset/debug.csv"
    # validationset="../dataset/debug.csv"
    testset="../dataset/debug.csv"

    ## Load Dataset
    print("Reading dataset...")
    reader_encoded = ipinyouReaderWithEncoding()
    trainDF, validateDF, testDF = reader_encoded.getTrainValidationTestDF_V2(trainset, validationset, testset)
    print(validateDF.shape)


    Y_column = 'click'
    # X_column = ['weekday', 'hour', 'useragent', 'region', 'city', 'domain', 'adexchange',
    #             'slotwidth', 'slotheight', 'slotprice', 'slotvisibility','slotformat', 'advertiser']
    # X_column = ['weekday', 'hour', 'useragent', 'region', 'city', 'domain', 'adexchange',
    #             'mobileos', 'slotdimension', 'slotprice', 'slotvisibility',
    #             'slotformat', 'advertiser']

    X_column = list(trainDF)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    print("X_column: ", X_column)



    click_pred = XGBoostBidModel(X_column, Y_column)
    # click_pred.gridSearch(trainDF)
    click_pred.trainModel(trainDF)
    click_pred.validateModel(validateDF)
    # click_pred.tunelinearBaseBid(validateDF)
    # click_pred.getBidPrice(validateDF)





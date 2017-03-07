from abc import ABCMeta, abstractmethod
from collections import defaultdict
import pandas as pd
import numpy as np
from Evaluator import Evaluator
import time
from sklearn.externals import joblib

class BidModelInterface():
    @abstractmethod
    def getBidPrice(self, oneBidRequest):
        raise NotImplementedError

    @abstractmethod
    def trainModel(self, allTrainData, retrain=True):
        raise NotImplementedError

    def saveModel(self, model, filename):
        """
        Save a scikit-learn trained model into a dumped file
        :param model:
        :param filename:
        :return:
        """
        print("Dumping trained model to file: ", filename)
        joblib.dump(self._model, filename)

    def loadSavedModel(self, filename):
        """
        Load a scikit-learn trained model from a previously dumped file
        :param filename:
        :return:
        """
        print("Loading trained model from file: ", filename)
        return joblib.load(filename)


class ConstantBidModel(BidModelInterface):
    def __init__(self, defaultbid=300):
        """
        Init the model
        Default bid is a hyperparameter from previous training

        :param defaultbid: Hyperparameter from previous training
        """
        # Based on training set
        # bestBid: 300
        # bestCTR: 0.000753964988446
        self.defaultBid = defaultbid

    def getBidPrice(self, allBidid):
        """
        Takes in all array of bidid and append a constant bid for everyone

        :param allBidid: Array of bidid
        :return: [bidid, bidprice]
        """
        bidprice = np.full(allBidid.shape[0], self.defaultBid, dtype=int)
        bids = np.stack([allBidid, bidprice], axis=1)

        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

        return bids

    def trainModel(self, allTrainData, searchRange=[1, 300], budget=25000*1000):
        """
        Train the constant model.

        A best bid price will be returned

        Budget should be in chinese fen * 1000

        :param allTrainData: Training data in matrix
        :param searchRange: Search Grid in array for best bid price. [lowerbound, upperbound]
        :param budget: The budget to use. chinese fen * 1000
        :return: The best constant bid price that obtained the highest CTR

        """
        # goldlabel = np.copy(allTrainData)
        # goldlabel = np.delete(goldlabel, [0, 21, 22], axis=1)# remove 'click','bidprice','payprice'

        bestBid = 0
        bestCTR = 0
        # print(goldlabel.shape)
        for bid in range(searchRange[0], searchRange[1]):
        # for bid in range(1000, 1001):  # To test cutting back budget
            self.defaultBid = bid
            # start_time = time.time()
            bids = self.getBidPrice(allTrainData.bidid)
            # print('Metrics np.apply_along_axis time: {} seconds'.format(round(time.time() - start_time, 2)))
            # myEvaluator = Evaluator(budget, bids, allTrainData)
            myEvaluator = Evaluator()
            resultDict = myEvaluator.computePerformanceMetricsDF(budget, bids, allTrainData)

            if resultDict['won'] != 0:
                print("Constant bid: {} CTR: {}".format(self.defaultBid, resultDict['click'] / resultDict['won']))
            else:
                print("Constant bid: {} CTR: not computed as no. of won is 0".format(self.defaultBid))

            if resultDict['won'] != 0:
                currentCTR = resultDict['click'] / resultDict['won']
            else:
                continue

            if currentCTR > bestCTR:
                bestCTR = currentCTR
                bestBid = bid


        print("bestBid: ", bestBid)
        print("bestCTR: ", bestCTR)

        self.defaultBid = bestBid

        return  self.defaultBid


class GaussianRandomBidModel(BidModelInterface):
    """
    Perform Random bidding using Gaussian distribution based on mean and stdev of payprice compute from train set
    """

    def getBidPrice(self, allBidid):
        # print("bid: ", oneBidRequest)
        bidprice = np.random.normal(loc=80.25102474739948, scale=6, size=len(allBidid))
        bids = np.stack([allBidid, bidprice], axis=1)
        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

        return bids

    def trainModel(self, allTrainData):
        raise NotImplementedError

class UniformRandomBidModel(BidModelInterface):
    """
    Perform Random bidding using uniform distribution in range of 0 to upper bound param for bid.
    """
    def __init__(self,bidupperbound=300):
        self.defaultBidUpperBound = bidupperbound

    def getBidPrice(self, allBidid):
        # print("bid: ", oneBidRequest)
        bidprice = np.random.uniform(low=0,high=self.defaultBidUpperBound, size=len(allBidid))
        bids = np.stack([allBidid, bidprice], axis=1)
        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

        return bids

    def trainModel(self, allTrainData):
        raise NotImplementedError
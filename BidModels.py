from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy as np
from Evaluator import Evaluator

class BidModelInterface():
    @abstractmethod
    def getBidPrice(self, oneBidRequest):
        raise NotImplementedError

    @abstractmethod
    def trainModel(self, allTrainData):
        raise NotImplementedError

    def computeBidPrice(self, basebid, pCTR, avgCTR):
        """
        The default computation to compute bid price
        The implemented model should have its own ways to gather the necessary parameters as follows
        :param basebid:
        :param pCTR:
        :param avgCTR:
        :return: bid
        """
        bid=basebid*(pCTR/avgCTR)
        return bid

class ConstantBidModel(BidModelInterface):
    def __init__(self, defaultbid=10):
        self.defaultBid = defaultbid # CTR(train): 0.00045433893684688776

    def getBidPrice(self, oneBidRequest):
        # print("bid: ", oneBidRequest)
        return [oneBidRequest[2], int(self.defaultBid)]# oneBidRequest[2] = bidid

    def trainModel(self, allTrainData, searchRange=[1, 300], budget=25000*1000):
        goldlabel = np.copy(allTrainData)
        goldlabel = np.delete(goldlabel, [0, 21, 22], axis=1)# remove 'click','bidprice','payprice'

        bestBid = 0
        bestCTR = 0
        # print(goldlabel.shape)
        for bid in range(searchRange[0], searchRange[1]):
        #for bid in range(1000, 1001):  # To test cutting back budget
            self.defaultBid = bid
            start_time = time.time()
            bids = np.apply_along_axis(self.getBidPrice, axis=1, arr=goldlabel) #TODO: this is also slow as unnessariily retrieving 1 at a time
            print('Metrics np.apply_along_axis time: {} seconds'.format(round(time.time() - start_time, 2)))
            myEvaluator = Evaluator(budget, bids, allTrainData)
            resultDict = myEvaluator.computePerformanceMetrics()
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
        # return a fake default first
        self.defaultBid = bestBid


class GaussianRandomBidModel(BidModelInterface):
    """
    Perform Random bidding using Gaussian distribution based on mean and stdev of payprice compute from train set
    """

    def getBidPrice(self, oneBidRequest):
        # print("bid: ", oneBidRequest)
        bid = np.random.normal(loc=80.25102474739948, scale=6)
        return np.array([oneBidRequest[2], int(bid)])

    def trainModel(self, allTrainData):
        raise NotImplementedError

class UniformRandomBidModel(BidModelInterface):
    """
    Perform Random bidding using uniform distribution in range of 0 to upper bound param for bid.
    """
    def __init__(self,bidupperbound=300):
        self.defaultBidUpperBound = bidupperbound

    def getBidPrice(self, oneBidRequest):
        # print("bid: ", oneBidRequest)
        bid = np.random.uniform(low=0,high=self.defaultBidUpperBound)
        return [oneBidRequest[2], int(bid)]

    def trainModel(self, allTrainData):
        raise NotImplementedError
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


class ConstantBidModel(BidModelInterface):
    def __init__(self):
        self.defaultBid = 10 # CTR(train): 0.00045433893684688776

    def getBidPrice(self, oneBidRequest):
        # print("bid: ", oneBidRequest)
        return [oneBidRequest[2], int(self.defaultBid)]# oneBidRequest[2] = bidid

    def trainModel(self, allTrainData):
        goldlabel = np.copy(allTrainData)
        goldlabel = np.delete(goldlabel, [0, 21, 22], axis=1)# remove 'click','bidprice','payprice'

        bestBid = 0
        bestCTR = 0
        # print(goldlabel.shape)
        for bid in range(1, 300):# TODO: this could be input param for the range perhaps
        #for bid in range(1000, 1001):  # To test cutting back budget
            self.defaultBid = bid
            bids = np.apply_along_axis(self.getBidPrice, axis=1, arr=goldlabel) #TODO: this is also slow as unnessariily retrieving 1 at a time
            myEvaluator = Evaluator(25000*1000, bids, allTrainData) #TODO: wouldn't train budget be different, i.e factor into account how many more entries there are?
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
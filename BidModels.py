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
        return [oneBidRequest[2], self.defaultBid]

    def trainModel(self, allTrainData):
        goldlabel = np.copy(allTrainData)
        goldlabel = np.delete(goldlabel, [0, 21, 22], axis=1)

        bestBid = 0
        bestCTR = 0
        # print(goldlabel.shape)
        for bid in range(200, 300):
            self.defaultBid = bid
            bids = np.apply_along_axis(self.getBidPrice, axis=1, arr=goldlabel)
            myEvaluator = Evaluator(25000, bids, allTrainData)
            resultDict = myEvaluator.computePerformanceMetrics()
            if resultDict['won'] != 0:
                print("CTR: ", resultDict['click'] / resultDict['won'])
            else:
                print("CTR not computed as no. of won is 0")

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
import ipinyouReader
from collections import defaultdict
import time
import numpy as np
from sklearn import metrics

class Evaluator():
    # def __init__(self, budget, ourBids, goldlabels):
    #     self.budget = budget
    #     self.ourBids = ourBids
    #     self.goldlabels = goldlabels

    # def computePerformanceMetrics(self):
    #     print("************************************************")
    #     print("Try not to use this. Use computePerformanceMetricsDF(self, budget, ourBidsDF, goldlabelsDF) instead")
    #     print("************************************************")
    #     self.resultDict = defaultdict(float)
    #
    #     # start_time = time.time()
    #     # resultIndex = 0
    #     # for i in range(self.goldlabels.shape[0]):
    #     #     # Check if bid id match?
    #     #     if self.ourBids[resultIndex][0] == self.goldlabels[i][3]:
    #     #
    #     #         # Check if bid price is greater than (or eq) gold's payprice - TA said it's greater or eq to payprice = win
    #     #         if int(self.ourBids[resultIndex][1]) >= int(self.goldlabels[i][22]):
    #     #             # Won the bid!!
    #     #             self.resultDict['won'] += 1
    #     #             # Add the pay price for this ad. Don't Div by 1000 to convert to Chinese fen #budget is x 1000 already
    #     #             self.resultDict['spend'] += (self.goldlabels[i][22])
    #     #
    #     #             # Check if gold is clicked
    #     #             if int(self.goldlabels[i][0]) == 1:
    #     #                 # Increment click counter
    #     #                 self.resultDict['click'] += 1
    #     #
    #     #         resultIndex += 1
    #     #
    #     #     # Stop when reach end of our bids or budget are used
    #     #     if resultIndex == self.ourBids.shape[0] or self.resultDict['spend'] >= self.budget:
    #     #         break
    #     #
    #     #
    #     # #print("Result Matched: ", resultIndex, " out of ", self.ourBids.shape[0])
    #     # print('Metrics compute time: {} seconds'.format(round(time.time() - start_time, 2)))
    #
    #     start_time = time.time()
    #     bidids_match=np.all(np.equal(self.ourBids[:, 0], self.goldlabels[:, 3])) #safety check bidids match
    #     if bidids_match:
    #         #if bid price is greater than (or eq) gold's payprice - TA said it's greater or eq to payprice = win
    #         wonbid=np.greater_equal(self.ourBids[:,1].astype(int),self.goldlabels[:,22])
    #         spend=self.goldlabels[wonbid,22] #use wonbid as mask
    #         clicks=self.goldlabels[wonbid,0] #use wonbid as mask
    #
    #         #work backwards until we are within budget
    #         i=-1
    #         while sum(spend) > self.budget:
    #             #print("{},{},{}".format(i,sum(spend),self.budget))
    #             spend[i] = 0
    #             clicks[i] = 0
    #             wonbid[i] = False
    #             i+=-1
    #
    #         self.resultDict={'won':sum(wonbid),'click':sum(clicks),'spend':sum(spend)} #TODO: verify spend consistency
    #     else:
    #         print("Bid Ids did not match in arrays")
    #
    #     print('Metrics compute time: {} seconds'.format(round(time.time() - start_time, 2)))
    #     return self.resultDict

    def computePerformanceMetricsDF(self, budget, ourBidsDF, goldlabelsDF,verbose=False):

        self.resultDict = defaultdict(float)

        start_time = time.time()
        bidids_match=np.all(np.equal(ourBidsDF.bidid, goldlabelsDF['bidid'])) #safety check bidids match
        if bidids_match:
            #if bid price is greater than (or eq) gold's payprice - TA said it's greater or eq to payprice = win
            wonbid=np.greater_equal(ourBidsDF['bidprice'].astype(int), goldlabelsDF['payprice'])
            wonbidIndex = np.where(wonbid)
            spend = goldlabelsDF['payprice'].iloc[wonbidIndex]
            clicks = goldlabelsDF.click.iloc[wonbidIndex]

            # work backwards until we are within budget
            i = -1
            overspend = sum(spend) - budget
            while overspend > 0 and len(spend) + i > 0:
                if verbose:
                    print("{},{},{}".format(i, sum(spend), budget))
                overspend += -spend.iloc[i]
                spend.iloc[i] = 0
                clicks.iloc[i] = 0
                wonbid.iloc[i] = False
                i += -1


            self.resultDict={'won':sum(wonbid),'click':sum(clicks),'spend':sum(spend),'trimmed_bids':-i-1}
        else:
            print("Bid Ids did not match in arrays")

        print('Metrics compute time: {} seconds'.format(round(time.time() - start_time, 2)))
        return self.resultDict

    def printResult(self):
        print("Trimmed Bids:",self.resultDict['trimmed_bids'])
        print("Won: ", self.resultDict['won'])
        print("Click: ", self.resultDict['click'])
        if self.resultDict['won'] != 0:
            print("CTR: ", self.resultDict['click'] / self.resultDict['won'])
            print("CPM: ", self.resultDict['spend']/(self.resultDict['won']/1000))
        else:
            print("CTR and CPM not computed as no. of won is 0")
        print("Spend: ", self.resultDict['spend'])
        if self.resultDict['click'] != 0:
            print("Average CPC: ", self.resultDict['spend'] / self.resultDict['click'])
        else:
            print("Average CPC not computed as click is 0")



# BUDGET = 25000
#
# bidReader = ipinyouReader.ipinyouReader("result.csv", header=None)
# validationFileReader = ipinyouReader.ipinyouReader("../dataset/validation.csv")
#
# ourBids = bidReader.getResult()
# goldlabel = validationFileReader.getTrainData()
#
# myEvaluator = Evaluator(BUDGET, ourBids, goldlabel)
# myEvaluator.computePerformanceMetrics()
# myEvaluator.printResult()

class ClickEvaluator():
    def printClickPredictionScore(self, y_Pred, y_Gold):
        '''
        Compute the Precision, Recall and F1 score:

        :param y_Pred: Predicted Result
        :param y_Gold: Gold Label
        :return: precision=pCTR, recall, f1==> for click=1
        '''

        print("Number of 1 in pred: ", np.count_nonzero(y_Pred))
        # y_Pred = np.array(y_Pred)
        # print(y_Pred.shape)
        # print(y_Gold.shape)
        # clicked = np.count_nonzero(np.logical_and(y_Pred == 1, y_Gold == 1))
        # print("Number of 1 when pred==Gold: ", clicked)
        print("Number of 0 in pred: ", len(y_Pred) - np.count_nonzero(y_Pred))

        print("Number of 1 in gold: ", np.count_nonzero(y_Gold))
        print("Number of 0 in gold: ", len(y_Gold) - np.count_nonzero(y_Gold))


        # print("pCTR = (clicked / 1 in pred): ", clicked/np.count_nonzero(y_Pred))
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_Gold, y_Pred)

        for i in range(len(p)):
            print("Click=%d \tPrecision\pCTR: %5.3f \tRecall: %5.3f \tF1: %5.3f" % (i, p[i], r[i], f1[i]))

        # print("Focus on click=1 recall score and Number of 1 in pred.")

        return p[i], r[i], f1[i]#, (clicked/np.count_nonzero(y_Pred))
import ipinyouReader
from collections import defaultdict


class Evaluator():
    def __init__(self, budget, ourBids, goldlabels):
        self.budget = budget
        self.ourBids = ourBids
        self.goldlabels = goldlabels

    def computePerformanceMetrics(self):

        self.resultDict = defaultdict(float)

        # start_time = time.time()
        # resultIndex = 0
        # for i in range(self.goldlabels.shape[0]):
        #     # Check if bid id match?
        #     if self.ourBids[resultIndex][0] == self.goldlabels[i][3]:
        #
        #         # Check if bid price is greater than (or eq) gold's payprice - TA said it's greater or eq to payprice = win
        #         if int(self.ourBids[resultIndex][1]) >= int(self.goldlabels[i][22]):
        #             # Won the bid!!
        #             self.resultDict['won'] += 1
        #             # Add the pay price for this ad. Don't Div by 1000 to convert to Chinese fen #budget is x 1000 already
        #             self.resultDict['spend'] += (self.goldlabels[i][22])
        #
        #             # Check if gold is clicked
        #             if int(self.goldlabels[i][0]) == 1:
        #                 # Increment click counter
        #                 self.resultDict['click'] += 1
        #
        #         resultIndex += 1
        #
        #     # Stop when reach end of our bids or budget are used
        #     if resultIndex == self.ourBids.shape[0] or self.resultDict['spend'] >= self.budget:
        #         break
        #
        #
        # #print("Result Matched: ", resultIndex, " out of ", self.ourBids.shape[0])
        # print('Metrics compute time: {} seconds'.format(round(time.time() - start_time, 2)))

        start_time = time.time()
        bidids_match=np.all(np.equal(self.ourBids[:, 0], self.goldlabels[:, 3])) #safety check bidids match
        if bidids_match:
            #if bid price is greater than (or eq) gold's payprice - TA said it's greater or eq to payprice = win
            wonbid=np.greater_equal(self.ourBids[:,1].astype(int),self.goldlabels[:,22])
            spend=self.goldlabels[wonbid,22] #use wonbid as mask
            clicks=self.goldlabels[wonbid,0] #use wonbid as mask

            #work backwards until we are within budget
            i=-1
            while sum(spend) > self.budget:
                #print("{},{},{}".format(i,sum(spend),self.budget))
                spend[i] = 0
                clicks[i] = 0
                wonbid[i] = False
                i+=-1

            self.resultDict={'won':sum(wonbid),'click':sum(clicks),'spend':sum(spend)} #TODO: verify spend consistency
        else:
            print("Bid Ids did not match in arrays")

        print('Metrics compute time: {} seconds'.format(round(time.time() - start_time, 2)))
        return self.resultDict

    def printResult(self):
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


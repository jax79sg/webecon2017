import ipinyouReader
from collections import defaultdict


class Evaluator():
    def __init__(self, budget, ourBids, goldlabels):
        self.budget = budget
        self.ourBids = ourBids
        self.goldlabels = goldlabels

    def computePerformanceMetrics(self):
        resultIndex = 0
        self.resultDict = defaultdict(float)

        for i in range(self.goldlabels.shape[0]):
            # Check if bid id match?
            if self.ourBids[resultIndex][0] == self.goldlabels[i][3]:

                # Check if bid price is greater than gold's payprice
                if int(self.ourBids[resultIndex][1]) > int(self.goldlabels[i][22]):
                    # Won the bid!!
                    self.resultDict['won'] += 1
                    # Add the pay price for this ad. Div by 1000 to convert to Chinese fen
                    self.resultDict['spend'] += (self.goldlabels[i][22]/1000)

                    # Check if gold is clicked
                    if int(self.goldlabels[i][0]) == 1:
                        # Increment click counter
                        self.resultDict['click'] += 1

                resultIndex += 1

            # Stop when reach end of our bids or budget are used
            if resultIndex == self.ourBids.shape[0] or self.resultDict['spend'] >= self.budget:
                break

        # print("Result Matched: ", resultIndex, " out of ", self.ourBids.shape[0])
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


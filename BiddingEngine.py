import ipinyouReader
import ipinyouWriter
import Evaluator
import BidModels
import numpy as np


def exeConstantBidModel(validationData, trainData=None, writeResult2CSV=False):
    # Constant Bidding Model
    constantBidModel = BidModels.ConstantBidModel()
    if trainData != None:
        constantBidModel.trainModel(trainData.getTrainData())
    bids = np.apply_along_axis(constantBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("result.csv", bids)
    myEvaluator = Evaluator.Evaluator(25000, bids, validationData.getTrainData())
    myEvaluator.computePerformanceMetrics()
    myEvaluator.printResult()



# # Read in train.csv to train the model
trainReader = ipinyouReader.ipinyouReader("../dataset/train.csv")
# trainData = trainReader.getTrainData()

# Read in Validation.csv for developmental testing
devReader = ipinyouReader.ipinyouReader("../dataset/validation.csv")
# devData = devReader.getTestData()

# Execute Constant Bid Model
exeConstantBidModel(validationData=devReader, trainData=trainReader, writeResult2CSV=False)




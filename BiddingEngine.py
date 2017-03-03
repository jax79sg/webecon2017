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
    myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    myEvaluator.computePerformanceMetrics()
    myEvaluator.printResult()

def exeGaussianRandomBidModel(validationData, trainData=None, writeResult2CSV=False):
    # gaussian random Bidding Model
    randomBidModel = BidModels.GaussianRandomBidModel()

    bids = np.apply_along_axis(randomBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("result.csv", bids)
    myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    myEvaluator.computePerformanceMetrics()
    myEvaluator.printResult()

def exeUniformRandomBidModel(validationData, trainData=None, writeResult2CSV=False):
    # uniform random Bidding Model
    randomBidModel = BidModels.UniformRandomBidModel(300) #upper bound for random bidding range
    # TODO: could train this too in a range.

    bids = np.apply_along_axis(randomBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("result.csv", bids)
    myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    myEvaluator.computePerformanceMetrics()
    myEvaluator.printResult()

# # Read in train.csv to train the model
trainReader = ipinyouReader.ipinyouReader("../dataset/train.csv")
#trainData = trainReader.getTrainData()

# Read in Validation.csv for developmental testing
devReader = ipinyouReader.ipinyouReader("../dataset/validation.csv")
#devData = devReader.getTestData()

# Execute Constant Bid Model
print("== Constant bid model")
exeConstantBidModel(validationData=devReader, trainData=trainReader, writeResult2CSV=False)

# Execute Gaussian Random Bid Model
print("== Gaussian random bid model")
exeGaussianRandomBidModel(validationData=devReader, trainData=trainReader, writeResult2CSV=False)

# Execute Uniform Random Bid Model
print("== Uniform random bid model")
exeUniformRandomBidModel(validationData=devReader, trainData=trainReader, writeResult2CSV=False)



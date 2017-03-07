import ipinyouReader
import ipinyouWriter
import Evaluator
import BidModels
import numpy as np
import LinearBidModel


def exeConstantBidModel(validationData, trainData=None, train=False, writeResult2CSV=False):
    # Constant Bidding Model
    constantBidModel = BidModels.ConstantBidModel()

    if train:
        constantBidModel.trainModel(trainData, searchRange=[1, 400], budget=int(25000*1000*8.88))

    bids = constantBidModel.getBidPrice(validationData.bidid)
    # bids = np.apply_along_axis(constantBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultConstantBidModel.csv", bids)
    # myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    # myEvaluator.computePerformanceMetrics()
    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validationData)
    myEvaluator.printResult()

def exeGaussianRandomBidModel(validationData, trainData=None, writeResult2CSV=False):
    # gaussian random Bidding Model
    randomBidModel = BidModels.GaussianRandomBidModel()

    bids = randomBidModel.getBidPrice(validationData.bidid)
    # bids = np.apply_along_axis(randomBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultGaussianRandomBidModel.csv", bids)
    # myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    # myEvaluator.computePerformanceMetrics()
    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validationData)
    myEvaluator.printResult()

def exeUniformRandomBidModel(validationData, trainData=None, writeResult2CSV=False):
    # uniform random Bidding Model
    randomBidModel = BidModels.UniformRandomBidModel(300) #upper bound for random bidding range
    # TODO: could train this too in a range.

    bids = randomBidModel.getBidPrice(validationData.bidid)
    # bids = np.apply_along_axis(randomBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultUniformRandomBidModel.csv", bids)
    # myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    # myEvaluator.computePerformanceMetrics()
    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validationData)
    myEvaluator.printResult()

def exeLogisticRegressionBidModel(validationData=None, trainData=None, writeResult2CSV=False):
    # Get regressionFormulaX
    X_column = list(trainData)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    final_x = X_column[0]
    for i in range(1, len(X_column)):
        final_x = final_x + ' + ' + X_column[i]

    lrBidModel = LinearBidModel.LinearBidModel(regressionFormulaY='click',
                                               regressionFormulaX=final_x,
                                               cBudget=272.412385 * 1000, avgCTR=0.2, modelType='logisticregression')
    print(type(validationData))
    lrBidModel.trainModel(trainData, retrain=True, modelFile="LogisticRegression.pkl")
    # lrBidModel.gridSearchandCrossValidate(trainData.getDataFrame())

    bids = lrBidModel.getBidPrice(validationData)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("LRbidModelresult.csv", bids)
    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000*1000, bids, validationData)
    myEvaluator.printResult()

def exeSGDBidModel(validationData=None, trainData=None, writeResult2CSV=False):
    # Get regressionFormulaX
    X_column = list(trainData)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    final_x = X_column[0]
    for i in range(1, len(X_column)):
        final_x = final_x + ' + ' + X_column[i]

    lrBidModel=LinearBidModel.LinearBidModel(regressionFormulaY='click', regressionFormulaX=final_x, cBudget=272.412385 * 1000, avgCTR=0.2, modelType='sgdclassifier')
    print(type(validationData))
    lrBidModel.trainModel(trainData, retrain=True, modelFile="SGDClassifier.pkl")
    # lrBidModel.gridSearchandCrossValidate(trainData.getDataFrame())

    bids = lrBidModel.getBidPrice(validationData)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("SGDbidModelresult.csv", bids)
    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000*1000, bids, validationData)
    myEvaluator.printResult()



# Read in train.csv to train the model
# trainset = "../dataset/debug.csv"
# validationset = "../dataset/debug.csv"
trainset = "../dataset/train_cleaned_prune.csv"
validationset = "../dataset/validation_cleaned_prune.csv"
testset = "../dataset/test.csv"

print("Reading dataset...")
reader_encoded = ipinyouReader.ipinyouReaderWithEncoding()
trainDF, validateDF, testDF = reader_encoded.getTrainValidationTestDF_V2(trainset, validationset, testset)

# TODO Make Constant model take in DF
# Execute Constant Bid Model
print("== Constant bid model")
exeConstantBidModel(validationData=validateDF, trainData=trainDF, train=True, writeResult2CSV=True)

# Execute Gaussian Random Bid Model
print("== Gaussian random bid model")
exeGaussianRandomBidModel(validationData=validateDF, trainData=None, writeResult2CSV=False)

# Execute Uniform Random Bid Model
print("== Uniform random bid model")
exeUniformRandomBidModel(validationData=validateDF, trainData=None, writeResult2CSV=False)

# Execute LR Bid Model
print("============ Logistic Regression bid model")
exeLogisticRegressionBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=True)

# Execute SDG Bid Model
print("============ SGD bid model")
exeSGDBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=True)
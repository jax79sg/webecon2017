import ipinyouReader
import ipinyouWriter
import Evaluator
import BidModels
import LinearBidModel
import FMBidModel
import pandas as pd
import numpy as np
from XGBoostBidModel import XGBoostBidModel
from LinearBidModel_v2 import LinearBidModel_v2
from BidPriceEstimator import BidEstimator

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

def exeXGBoostBidModel(validationData, trainData=None, writeResult2CSV=False):
    Y_column = 'click'
    X_column = list(trainDF)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]

    xgd = XGBoostBidModel(X_column, Y_column)
    xgd.trainModel(trainData)
    bids = xgd.getBidPrice(validationData)

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultXGBoostBidModel.csv", bids)

    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validationData)
    myEvaluator.printResult()

    return xgd.getY_Pred(validationData)


def exeLogisticRegressionBidModel(validationData=None, trainData=None, writeResult2CSV=False):
    # Get regressionFormulaX
    X_column = list(trainData)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    final_x = X_column[0]
    for i in range(1, len(X_column)):
        final_x = final_x + ' + ' + X_column[i]

    lrBidModel = LinearBidModel.LinearBidModel(regressionFormulaY='click', regressionFormulaX=final_x, cBudget=272.412385 * 1000, avgCTR=0.2, modelType='logisticregression')
    print(type(validationData))
    lrBidModel.trainModel(trainData, retrain=True, modelFile="LogisticRegression.pkl")
    # lrBidModel.gridSearchandCrossValidate(trainData.getDataFrame())

    bids = lrBidModel.getBidPrice(validationData)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("LRbidModelresult.csv", bids)
    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000*1000, bids, validationData)
    myEvaluator.printResult()

def exeLogisticRegressionBidModel_v2(validationReader=None, trainReader=None, writeResult2CSV=False):
    trainOneHotData, trainY = trainReader.getOneHotData()
    validationOneHotData, valY = validationReader.getOneHotData(
        train_cols=trainOneHotData.columns.get_values().tolist())

    X_train = trainOneHotData
    Y_train = trainY['click']
    X_val = validationOneHotData
    Y_val = valY['click']

    lbm = LinearBidModel_v2(cBudget=272.412385 * 1000, avgCTR=0.2)
    lbm.trainModel(X_train, Y_train)
    # lbm.gridSearchandCrossValidate(X_train, Y_train)
    # print (validationReader.getDataFrame().info())
    v_df = validationReader.getDataFrame()

    bids = lbm.getBidPrice(X_val, v_df)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultLogisticRegressionBidModel.csv", bids)

    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, v_df)
    myEvaluator.printResult()

def exeSGDBidModel(validationData=None, trainData=None, writeResult2CSV=False):
    # Get regressionFormulaX
    X_column = list(trainData)
    print("X_column:",X_column)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    final_x = X_column[0]
    for i in range(1, len(X_column)):
        final_x = final_x + ' + ' + X_column[i]

    print("regressionFormulaX:", final_x)
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

def exeFM_ALSBidModel(validationDataOneHot=None, trainDataOneHot=None, validationData=None, writeResult2CSV=False):
    # Get regressionFormulaX
    X_column = list(trainDataOneHot)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    final_x = X_column[0]
    for i in range(1, len(X_column)):
        final_x = final_x + ' + ' + X_column[i]

    fmBidModel=FMBidModel.FMBidModel(regressionFormulaY='click', regressionFormulaX=X_column, cBudget=272.412385 * 1000, avgCTR=0.2, modelType='fmclassificationals')
    fmBidModel.trainModel(trainDataOneHot, retrain=True, modelFile="FMALSClassifier.pkl")
    fmBidModel.validateModel(validationDataOneHot, validationData)
    # lrBidModel.gridSearchandCrossValidate(trainData.getDataFrame())

    bids = fmBidModel.getBidPrice(validationDataOneHot)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("FMALSbidModelresult.csv", bids)
    myEvaluator = Evaluator.Evaluator()

    #Convert back to label 1 and 0 for evaluator
    validationData['click'] = validationData['click'].map({-1: 0, 1: 1})
    myEvaluator.computePerformanceMetricsDF(25000*1000, bids, validationData)
    myEvaluator.printResult()

def exeFM_SGDBidModel(validationDataOneHot=None, trainDataOneHot=None, validationData=None, writeResult2CSV=False):
    # Get regressionFormulaX
    X_column = list(trainDataOneHot)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]
    final_x = X_column[0]
    for i in range(1, len(X_column)):
        final_x = final_x + ' + ' + X_column[i]
    #TODO: FM with SGD is currently predicting all click as 0. Training was also stuck at click=0. Will need to analyse and fix.
    fmBidModel=FMBidModel.FMBidModel(regressionFormulaY='click', regressionFormulaX=X_column, cBudget=272.412385 * 1000, avgCTR=0.2, modelType='fmclassificationsgd')
    fmBidModel.trainModel(trainDataOneHot, retrain=True, modelFile="FMSGDClassifier.pkl")
    fmBidModel.validateModel(validationDataOneHot, validationData)
    # lrBidModel.gridSearchandCrossValidate(trainData.getDataFrame())

    bids = fmBidModel.getBidPrice(validationDataOneHot)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("FMSGDbidModelresult.csv", bids)
    myEvaluator = Evaluator.Evaluator()

    #Convert back to label 1 and 0 for evaluator
    validationData['click'] = validationData['click'].map({-1: 0, 1: 1})
    myEvaluator.computePerformanceMetricsDF(25000*1000, bids, validationData)
    myEvaluator.printResult()

def exeEnsemble_v1(validationData):
    xg_y_pred = exeXGBoostBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=False)
    cnn_y_pred, slotprices = exeCNNBidModel(...)
    # fm_y_pred = exeFM_SGDBidModel(validationDataOneHot=validateDFonehot, trainDataOneHot=trainDFonehot, validationData=validateDF, writeResult2CSV=True)

    # Use XG's 0 when its threshold is below 0.75.
    y_pred = [0 if xg<0.75 else cnn for xg, cnn in zip(xg_y_pred, cnn_y_pred)]

    # Use CNN's 1 when its threshold is above 0.2?
    prune_thresh = 0.2

    be = BidEstimator()
    bidprice = be.linearBidPrice_mConfi(y_pred, 230, 100, prune_thresh)
    # bidprice = be.linearBidPrice_variation(y_pred, 80, 0.2, slotprices=slotprices, prune_thresh=prune_thresh)
    bids = np.stack([validationData['bidid'], bidprice], axis=1)
    bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(25000*1000, bids, validationData)

    # Force CNN result to 1 and 0 for F1 score
    y_pred = [1 if i >= prune_thresh else 0 for i in y_pred]
    ce = Evaluator.ClickEvaluator()
    ce.printClickPredictionScore(y_pred, validationData)



# Read in train.csv to train the model
# trainset = "../dataset/debug.csv"
# validationset = "../dataset/debug.csv"
trainset = "../dataset/train_cleaned_prune.csv"
validationset = "../dataset/validation_cleaned_prune.csv"
testset = "../dataset/combined.csv"

print("Reading dataset...")
reader_encoded = ipinyouReader.ipinyouReaderWithEncoding()
trainDF, validateDF, testDF = reader_encoded.getTrainValidationTestDF_V2(trainset, validationset, testset)

trainReader = ipinyouReader.ipinyouReader(trainset)
validationReader = ipinyouReader.ipinyouReader(validationset)

# # Execute Constant Bid Model
# print("== Constant bid model")
# exeConstantBidModel(validationData=validateDF, trainData=trainDF, train=True, writeResult2CSV=True)
#
# # Execute Gaussian Random Bid Model
# print("== Gaussian random bid model")
# exeGaussianRandomBidModel(validationData=validateDF, trainData=None, writeResult2CSV=False)
#
# # Execute Uniform Random Bid Model
# print("== Uniform random bid model")
# exeUniformRandomBidModel(validationData=validateDF, trainData=None, writeResult2CSV=False)
#
# # Execute XGBoost Bid Model
# print("== XGBoost bid model")
# exeXGBoostBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=False)
#
# # Execute LR Bid Model
# print("============ Logistic Regression bid model")
# exeLogisticRegressionBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=True)
#
# Execute LR Bid Model (Use One-hot Encoding)
print("============ Logistic Regression bid model (Use One-hot Encoding)")
exeLogisticRegressionBidModel_v2(validationReader=validationReader, trainReader=trainReader, writeResult2CSV=True)
#
# # Execute SDG Bid Model
# print("============ SGD bid model")
# exeSGDBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=True)

# # Execute FM Bid Model
# print("============ Factorisation Machine bid model....setting up")
# combinedDF=testDF
# X_column = list(combinedDF)
# unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
# [X_column.remove(i) for i in unwanted_Column]
# final_x = X_column[0]
# for i in range(1, len(X_column)):
#     final_x = final_x + ' + ' + X_column[i]
#
# print("FastFM classification only accepts -1 and 1 as Gold labels. Changing gold labels from 0 to -1")
# combinedDF['click'] = combinedDF['click'].map({0: -1, 1: 1})
# validateDF['click'] = validateDF['click'].map({0: -1, 1: 1})
# # combinedDF.to_csv(path_or_buf="combinedDF.csv")
# print("Performing one hot encoding of combined set")
# combinedDF = pd.get_dummies(data=combinedDF,sparse=True, columns=X_column)
# print("combinedDF Cols:", list(combinedDF))
# print("Split back into train and val sets...gonna take a while")
# trainDFonehot=combinedDF.ix[0:9928]
# validateDFonehot=combinedDF.ix[9929:]
# # trainDF.to_csv(path_or_buf="trainDF.csv")
# # validateDF.to_csv(path_or_buf="validateDF.csv")
#
# print("============ FM ALS bid model")
# exeFM_ALSBidModel(validationDataOneHot=validateDFonehot, trainDataOneHot=trainDFonehot, validationData=validateDF, writeResult2CSV=True)
#
# print("============ FM SGD bid model")
# #No idea why validateDF  got mutated after calling exeFM_ALSBidModel, so have to transform back.
# validateDF['click'] = validateDF['click'].map({0: -1, 1: 1})
# exeFM_SGDBidModel(validationDataOneHot=validateDFonehot, trainDataOneHot=trainDFonehot, validationData=validateDF, writeResult2CSV=True)

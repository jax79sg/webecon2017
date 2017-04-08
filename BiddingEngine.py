import ipinyouReader
import ipinyouWriter
import Evaluator
import BidModels
import LinearBidModel
import pandas as pd
import numpy as np
from XGBoostBidModel import XGBoostBidModel
from LinearBidModel_v2 import LinearBidModel_v2
from BidPriceEstimator import BidEstimator
from CNNBidModel import *
from Utilities import Utility
# import FMBidModel

def exeConstantBidModel(validationData, trainData=None, train=False, writeResult2CSV=False):
    # Constant Bidding Model
    constantBidModel = BidModels.ConstantBidModel(defaultbid=77)

    if train:
        constantBidModel.trainModel(trainData, searchRange=[1, 300], budget=int(6250*1000*8.88))

    bids = constantBidModel.getBidPrice(validationData.bidid)
    # bids = np.apply_along_axis(constantBidModel.getBidPrice, axis=1, arr=validationData.getTestData())

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultConstantBidModel.csv", bids)
    # myEvaluator = Evaluator.Evaluator(25000*1000, bids, validationData.getTrainData())
    # myEvaluator.computePerformanceMetrics()

    myEvaluator = Evaluator()
    myEvaluator.computePerformanceMetricsDF(6250 * 1000, bids, validationData)
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
    myEvaluator = Evaluator()
    myEvaluator.computePerformanceMetricsDF(6250 * 1000, bids, validationData)
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
    myEvaluator = Evaluator()
    myEvaluator.computePerformanceMetricsDF(6250 * 1000, bids, validationData)
    myEvaluator.printResult()

def exeXGBoostBidModel(validationData, trainData=None, writeResult2CSV=False, testMode=True):
    Y_column = 'click'
    X_column = list(trainDF)
    unwanted_Column = ['click', 'bidid', 'bidprice', 'payprice', 'userid', 'IP', 'url', 'creative', 'keypage']
    [X_column.remove(i) for i in unwanted_Column]

    xgd = XGBoostBidModel(X_column, Y_column)
    xgd.trainModel(trainData)
    bids = xgd.getBidPrice(validationData)

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultXGBoostBidModel.csv", bids)

    if not testMode:
        myEvaluator = Evaluator()
        myEvaluator.computePerformanceMetricsDF(6250 * 1000, bids, validationData)
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
    myEvaluator.computePerformanceMetricsDF(6250*1000, bids, validationData)
    myEvaluator.printResult()

def exeLogisticRegressionBidModel_v2(validationReader=None, trainReader=None, writeResult2CSV=False):
    trainOneHotData, trainY = trainReader.getOneHotData()
    validationOneHotData, valY = validationReader.getOneHotData(
        train_cols=trainOneHotData.columns.get_values().tolist())

    X_train = trainOneHotData
    Y_train = trainY['click']
    X_val = validationOneHotData
    Y_val = valY['click']

    lbm = LinearBidModel_v2(cBudget=110, avgCTR=0.2)
    lbm.trainModel(X_train, Y_train)
    # lbm.gridSearchandCrossValidate(X_train, Y_train)
    # print (validationReader.getDataFrame().info())
    v_df = validationReader.getDataFrame()

    y_pred, bids = lbm.getBidPrice(X_val, v_df)
    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultLogisticRegressionBidModel.csv", bids)

    myEvaluator = Evaluator()
    myEvaluator.computePerformanceMetricsDF(6250 * 1000, bids, v_df)
    myEvaluator.printResult()

    return y_pred


def exeFMBidModel(testDF=None, validateDF=None, trainDF=None, trainReader=None, validationReader=None, testReader=None, writeResult2CSV=False):
    print("============ Factorisation Machine bid model....setting up")

    timer = Utility()
    timer.startTimeTrack()

    print("Getting encoded datasets")
    trainOneHotData, trainY = trainReader.getOneHotData()
    validationOneHotData, valY = validationReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist())
    testOneHotData, testY = testReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist())
    timer.checkpointTimeTrack()

    print("trainOneHotData:",trainOneHotData.shape,list(trainOneHotData))
    print("trainY:", trainY.shape, list(trainY))
    print("validationOneHotData:",validationOneHotData.shape,list(validationOneHotData))
    print("valY:", valY.shape, list(valY))

    fmBidModel=FMBidModel.FMBidModel(cBudget=6250 * 1000, modelType='fmclassificationsgd')
    print("==========Training starts")
    # fmBidModel.gridSearchandCrossValidateFastSGD(trainOneHotData, trainY)
    # timer.checkpointTimeTrack()

    fmBidModel.trainModel(trainOneHotData,trainY, retrain=True, modelFile="data.pruned/fmclassificationsgd.pkl")
    timer.checkpointTimeTrack()

    print("==========Validation starts")
    predictedProb=fmBidModel.validateModel(validationOneHotData, valY)
    timer.checkpointTimeTrack()

    # print("==========Bid optimisation starts")
    # fmBidModel.optimiseBid(validationOneHotData,valY)
    # timer.checkpointTimeTrack()

    # best score      0.3683528286042599
    # noBidThreshold  2.833333e-01
    # minBid          2.000000e+02
    # bidRange        9.000000e+01
    # sigmoidDegree - 1.000000e+01
    # won             3.432900e+04
    # click           1.380000e+02
    # spend           2.729869e+06
    # trimmed_bids    0.000000e+00
    # CTR             4.019925e-03
    # CPM             7.952078e+04
    # CPC             1.978166e+04
    # blended_score   3.683528e-01

    # best score      0.3681133881545131
    # noBidThreshold  2.833333e-01
    # minBid          2.000000e+02
    # bidRange        1.000000e+02
    # sigmoidDegree - 1.000000e+01
    # won             3.449900e+04
    # click           1.380000e+02
    # spend           2.758561e+06
    # trimmed_bids    0.000000e+00
    # CTR             4.000116e-03
    # CPM             7.996061e+04
    # CPC             1.998957e+04
    # blended_score   3.681134e-01


    # New budget      6250000
    # FM
    # best score      0.32755084132163526
    # noBidThreshold  8.666667e-01
    # minBid          2.000000e+02
    # bidRange        2.500000e+02
    # sigmoidDegree - 1.000000e+01
    # won             1.461000e+04
    # click           1.170000e+02
    # spend           1.124960e+06
    # trimmed_bids    0.000000e+00
    # CTR             8.008214e-03
    # CPM             7.699932e+04
    # CPC             9.615043e+03
    # blended_score   3.275508e-01

    # print("==========Getting  bids")
    ## 25000 budget
    # bidIdPriceDF=fmBidModel.getBidPrice(validationOneHotData,valY,noBidThreshold=0.2833333,minBid=200,bidRange=100,sigmoidDegree=-10)
    ## 6250 budget
    # bidIdPriceDF=fmBidModel.getBidPrice(validationOneHotData,valY,noBidThreshold=0.8666667,minBid=200,bidRange=250,sigmoidDegree=-10)
    # print("bidIdPriceDF:",bidIdPriceDF.shape, list(bidIdPriceDF))
    # bidIdPriceDF.to_csv("mybids.csv")
    # timer.checkpointTimeTrack()

    return predictedProb

def exeEnsemble_v1(trainDF, targetDF, trainPath, validationPath, targetPath, writeResult2CSV=False):
    xg_y_pred = exeXGBoostBidModel(validationData=targetDF, trainData=trainDF, writeResult2CSV=False)
    cnn_y_pred = exeCNNBidModel(validationDataPath=validationPath, trainDataPath=trainset, testDataPath=targetPath, writeResult2CSV=False)
    # fm_y_pred = exeFM_SGDBidModel(validationDataOneHot=validateDFonehot, trainDataOneHot=trainDFonehot, validationData=validateDF, writeResult2CSV=True)

    # Use XG's 0 when its threshold is below 0.75.
    y_pred = [0 if xg < 0.75 else cnn for xg, cnn in zip(xg_y_pred, cnn_y_pred)]

    # Use CNN's 1 when its threshold is above 0.2?
    prune_thresh = 0.2

    be = BidEstimator()
    bidprice = be.linearBidPrice_mConfi(y_pred, 230, 100, prune_thresh)
    # bidprice = be.linearBidPrice_variation(y_pred, 80, 0.2, slotprices=slotprices, prune_thresh=prune_thresh)
    bids = np.stack([targetDF['bidid'], bidprice], axis=1)
    bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

    if writeResult2CSV:
        ipinyouWriter.ResultWriter().writeResult("resultEnsemble_v1.csv", bids)

    myEvaluator = Evaluator.Evaluator()
    myEvaluator.computePerformanceMetricsDF(6250*1000, bids, targetDF)

    # Force CNN result to 1 and 0 for F1 score
    y_pred = [1 if i >= prune_thresh else 0 for i in y_pred]
    ce = Evaluator.ClickEvaluator()
    ce.printClickPredictionScore(y_pred, targetDF)

def exeEnsemble_v2(trainDF, validateDF, testDF,
                   trainPath, validationPath, testPath,
                   trainReader, validateReader, testReader,
                   writeResult2CSV=False):
    '''
    Takes the average of y_pred from all models.
    '''
    xg_y_pred = exeXGBoostBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=False)
    cnn_y_pred = exeCNNBidModel(validationDataPath=validationPath, trainDataPath=trainPath, testDataPath=testPath, writeResult2CSV=False)
    lr_y_pred = exeLogisticRegressionBidModel_v2(validationReader=validationReader, trainReader=trainReader, writeResult2CSV=False)
    fm_y_pred=exeFMBidModel(trainReader=trainReader, validationReader=validateReader, testReader=testReader, writeResult2CSV=False)

    # Average them
    # y_pred = [(xg+ lr) / 2.0 for xg, lr in zip(xg_y_pred, lr_y_pred)]
    # y_pred = [(xg + cnn + lr)/3.0 for xg, cnn, lr in zip(xg_y_pred, cnn_y_pred, lr_y_pred)]
    y_pred = [(xg + cnn + lr + fm) / 4.0 for xg, cnn, lr, fm in zip(xg_y_pred, cnn_y_pred, lr_y_pred, fm_y_pred)]


    print("XGBoost AUC:")
    ClickEvaluator().clickROC(validateDF['click'], xg_y_pred, False)
    print("CNN AUC:")
    ClickEvaluator().clickROC(validateDF['click'], cnn_y_pred, False)
    print("Logistic AUC:")
    ClickEvaluator().clickROC(validateDF['click'], lr_y_pred, False)
    print("FastFM AUC:")
    ClickEvaluator().clickROC(validateDF['click'], fm_y_pred, False)

    print("Ensemble AUC:")
    ClickEvaluator().clickROC(validateDF['click'], y_pred, False)

    y_pred = np.array(y_pred)
    click1 = y_pred[validateDF.click == 1]
    n, bins, patches = ClickEvaluator().clickProbHistogram(pred_prob=click1, color='g',
                                                           title='Predicted probabilities for clicks=1',
                                                           # imgpath="./SavedCNNModels/ensemblev2-click1-" + bidmodel.timestr + ".jpg",
                                                           showGraph=True)

    # click=0 prediction as click=1 probabilities
    click0 = y_pred[validateDF.click == 0]
    n, bins, patches = ClickEvaluator().clickProbHistogram(pred_prob=click0, color='r',
                                                           title='Predicted probabilities for clicks=0',
                                                           # imgpath="./SavedCNNModels/ensemblev2-click0-" + bidmodel.timestr + ".jpg",
                                                           showGraph=True)

def exeEnsemble_Weighted(trainDF, validateDF, testDF,
                   trainPath, validationPath, testPath,
                   trainReader, validateReader, testReader,
                   writeResult2CSV=False):
    '''
    Takes the average of y_pred from all models.
    '''
    xg_y_pred = exeXGBoostBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=False)
    cnn_y_pred = exeCNNBidModel(validationDataPath=validationPath, trainDataPath=trainPath, testDataPath=testPath, writeResult2CSV=False)
    lr_y_pred = exeLogisticRegressionBidModel_v2(validationReader=validationReader, trainReader=trainReader, writeResult2CSV=False)
    fm_y_pred=exeFMBidModel(trainReader=trainReader, validationReader=validateReader, testReader=testReader, writeResult2CSV=False)

    # Average them
    # y_pred = [(xg+ lr) / 2.0 for xg, lr in zip(xg_y_pred, lr_y_pred)]
    # y_pred = [(xg + cnn + lr)/3.0 for xg, cnn, lr in zip(xg_y_pred, cnn_y_pred, lr_y_pred)]
    y_pred = [(xg*0.4 + cnn*0.4 + lr*0.05 + fm*0.15)  for xg, cnn, lr, fm in zip(xg_y_pred, cnn_y_pred, lr_y_pred, fm_y_pred)]

    #This one hits 0.874 for the xg/lr/fm emsemble models, perviously 0.861 (Can't run CNN on my mac yet, got this convolution missing error)
    # y_pred = [(xg * 0.6 + lr * 0.1 + fm * 0.3) for xg, lr, fm in zip(xg_y_pred, lr_y_pred, fm_y_pred)]


    print("XGBoost AUC:")
    ClickEvaluator().clickROC(validateDF['click'], xg_y_pred, False)
    # print("CNN AUC:")
    # ClickEvaluator().clickROC(validateDF['click'], cnn_y_pred, False)
    print("Logistic AUC:")
    ClickEvaluator().clickROC(validateDF['click'], lr_y_pred, False)
    print("FastFM AUC:")
    ClickEvaluator().clickROC(validateDF['click'], fm_y_pred, False)

    print("Ensemble AUC:")
    ClickEvaluator().clickROC(validateDF['click'], y_pred, False)

    y_pred = np.array(y_pred)
    click1 = y_pred[validateDF.click == 1]
    n, bins, patches = ClickEvaluator().clickProbHistogram(pred_prob=click1, color='g',
                                                           title='Predicted probabilities for clicks=1',
                                                           # imgpath="./SavedCNNModels/ensemblev2-click1-" + bidmodel.timestr + ".jpg",
                                                           showGraph=True)

    # click=0 prediction as click=1 probabilities
    click0 = y_pred[validateDF.click == 0]
    n, bins, patches = ClickEvaluator().clickProbHistogram(pred_prob=click0, color='r',
                                                           title='Predicted probabilities for clicks=0',
                                                           # imgpath="./SavedCNNModels/ensemblev2-click0-" + bidmodel.timestr + ".jpg",
                                                           showGraph=True)



def exeCNNBidModel(validationDataPath, trainDataPath, testDataPath, writeResult2CSV=False):
    print("===== exeCNNBidModel start =====")

    TRAIN_FILE_PATH=trainDataPath
    VALIDATION_FILE_PATH = validationDataPath
    TEST_FILE_PATH = testDataPath
    SHUFFLE_INPUT = True
    RESERVE_VAL = True # Reserve validation set for further tuning, in CTR training just use train splits

    ### Weights
    CLASS_WEIGHTS_MU = 2.2  # 0.8 #0.15

    ### Features
    EXCLUDE_DOMAIN=False
    DOMAIN_KEEP_PROB=0.05 #1.0

    ### Training
    BATCH_SIZE = 32
    TOTAL_EPOCHS = 20
    LEARNING_RATE = 0.0001  # adam #for SGD 0.003

    ##########
    ## Load Dataset
    print("==== Reading in train set...")
    print("Train file: {}".format(TRAIN_FILE_PATH))
    trainReader = ipinyouReader.ipinyouReader(TRAIN_FILE_PATH)

    ## onehot
    print("== Convert to one-hot encoding...")
    trainOneHotData, trainY = trainReader.getOneHotData(exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
    print("Train set - No. of one-hot features: {}".format(len(trainOneHotData.columns)))

    print("==== Reading in validation set...")
    print("Validation file: {}".format(VALIDATION_FILE_PATH))
    validationReader = ipinyouReader.ipinyouReader(VALIDATION_FILE_PATH)
    valOneHotData, valY = validationReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
    print("Validation set - No. of one-hot features: {}".format(len(valOneHotData.columns)))

    print("==== Reading in test set...")
    print("Test file: {}".format(TEST_FILE_PATH))
    testReader = ipinyouReader.ipinyouReader(TEST_FILE_PATH)
    testOneHotData, testbidids = testReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),
                                                          exclude_domain=EXCLUDE_DOMAIN,
                                                          domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
    print("Test set - No. of one-hot features: {}".format(len(testOneHotData.columns)))

    print("==== Train CNN model...")
    bidmodel = CNNBidModel(trainOneHotData, trainY,valOneHotData,valY,testOneHotData,testbidids, class_weights_mu=CLASS_WEIGHTS_MU,batch_size=BATCH_SIZE, total_epochs=TOTAL_EPOCHS, learning_rate=LEARNING_RATE, shuffle=SHUFFLE_INPUT,reserve_val=RESERVE_VAL)
    bidmodel.trainModel()
    print("== Reload to best weights saved...")
    bidmodel.loadSavedModel(bidmodel.model_checkpoint_filepath)


    print("==== Predict clicks...")
    click_eval = ClickEvaluator()
    prob_click_train=bidmodel.predictClickProbs(bidmodel.X_train)
    if RESERVE_VAL:
        prob_click_val = bidmodel.predictClickProbs(bidmodel.X_val)

        print("== Click prob distributions...")
        # click=1 prediction as click = =1 probabilities
        click1 = prob_click_val[bidmodel.Y_click_val[:, 1].astype(bool), 1]
        n, bins, patches = click_eval.clickProbHistogram(pred_prob=click1,color='g',title='Predicted probabilities for clicks=1',imgpath="./SavedCNNModels/Keras-CNN-click1-" + bidmodel.timestr + ".jpg",showGraph=False)

        # click=0 prediction as click=1 probabilities
        click0 = prob_click_val[bidmodel.Y_click_val[:, 0].astype(bool), 1]
        n, bins, patches = click_eval.clickProbHistogram(pred_prob=click0,color='r',title='Predicted probabilities for clicks=0',imgpath="./SavedCNNModels/Keras-CNN-click0-" + bidmodel.timestr + ".jpg",showGraph=False)

        print("== ROC for click model...")
        roc_auc = click_eval.clickROC(bidmodel.Y_click_val[:, 1], prob_click_val[:, 1],imgpath="./SavedCNNModels/Keras-CNN-ROC-" + bidmodel.timestr + ".jpg",showGraph=False)

    print("===== exeCNNBidModel end =====")

    return prob_click_val[:,1]

# Read in train.csv to train the model
# trainset = "../dataset/train_cleaned.csv"
# validationset = "../dataset/debug.csv"
trainset = "./data.final/train1_cleaned_prune.csv"
validationset = "./data.final/validation_cleaned.csv"
testset = "./data.final/test.csv"

print("Reading dataset...")
# reader_encoded = ipinyouReader.ipinyouReaderWithEncoding()
# trainDF, validateDF, testDF = reader_encoded.getTrainValidationTestDF_V2(trainset, validationset, testset)

trainReader = ipinyouReader.ipinyouReader(trainset)
validationReader = ipinyouReader.ipinyouReader(validationset)
testReader = ipinyouReader.ipinyouReader(testset)

# # Execute Constant Bid Model
# print("== Constant bid model")
# exeConstantBidModel(validationData=validationReader.getDataFrame(), trainData=trainReader.getDataFrame(), train=False, writeResult2CSV=True)
#
# # # Execute Gaussian Random Bid Model
# print("== Gaussian random bid model")
# exeGaussianRandomBidModel(validationData=validationReader.getDataFrame(), trainData=None, writeResult2CSV=False)
#
# # Execute Uniform Random Bid Model
# print("== Uniform random bid model")
# exeUniformRandomBidModel(validationData=validationReader.getDataFrame(), trainData=None, writeResult2CSV=False)
#
# # Execute XGBoost Bid Model
# print("== XGBoost bid model")
# exeXGBoostBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=False)
#
# Execute CNN Bid Model
#print("== CNN bid model")
#yPred_CNNBidModel = exeCNNBidModel(validationDataPath=validationset, trainDataPath=trainset, testDataPath=testset, writeResult2CSV=False)
#print(yPred_CNNBidModel)

# # Execute LR Bid Model
# print("============ Logistic Regression bid model")
# exeLogisticRegressionBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=True)
#
# Execute LR Bid Model (Use One-hot Encoding)
# print("============ Logistic Regression bid model (Use One-hot Encoding)")
# exeLogisticRegressionBidModel_v2(validationReader=validationReader, trainReader=trainReader, writeResult2CSV=True)
#
# # Execute SDG Bid Model
# print("============ SGD bid model")
# exeSGDBidModel(validationData=validateDF, trainData=trainDF, writeResult2CSV=True)

# # Execute FM Bid Model
# print("============ FM ALS bid model")
# exeFM_ALSBidModel(testDF=testDF, validateDF=validateDF, trainDF=trainDF, writeResult2CSV=True)
#
# print("============ FM SGD bid model")
# #No idea why validateDF  got mutated after calling exeFM_ALSBidModel, so have to transform back.
# validateDF['click'] = validateDF['click'].map({0: -1, 1: 1})
# #exeFM_SGDBidModel(validationDataOneHot=validateDFonehot, trainDataOneHot=trainDFonehot, validationData=validateDF, writeResult2CSV=True)
# exeFM_SGDBidModel(testDF=testDF, validateDF=validateDF, trainDF=trainDF, writeResult2CSV=True)

# Execute Ensemble V2 Bid Model
# print("============ Ensemble V2 Bid Model")
# exeEnsemble_v2(trainDF, validateDF, testDF,
#                trainset, validationset, testset,
#                trainReader, validationReader, testReader,
#                writeResult2CSV=False)

# print("============ Ensemble V2 Bid Model")
# exeEnsemble_Weighted(trainDF, validateDF, testDF,
#                trainset, validationset, testset,
#                trainReader, validationReader, testReader,
#                writeResult2CSV=False)
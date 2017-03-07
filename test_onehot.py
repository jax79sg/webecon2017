import ipinyouReader
import ipinyouWriter
import Evaluator
import BidModels
import numpy as np
import LogisticRegressionBidModel

# # Read in train.csv to train the model
#trainReader = ipinyouReader.ipinyouReader("../dataset/train.csv")
trainReader = ipinyouReader.ipinyouReader("./data.pruned/train_cleaned_prune.csv")
#trainData = trainReader.getTrainData()

# Read in Validation.csv for developmental testing
#devReader = ipinyouReader.ipinyouReader("../dataset/validation.csv")
devReader = ipinyouReader.ipinyouReader("./data.pruned/validation_cleaned_prune.csv")
#devData = devReader.getTestData()

# onehot
trainOneHotData,trainY = trainReader.getOneHotData()

devOneHotData,devY = devReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist())

print(len(trainOneHotData.columns))
print(len(devOneHotData.columns))


X_train = trainOneHotData.as_matrix()
Y_train = trainY.as_matrix()
X_val = devOneHotData.as_matrix()
Y_val = devY.as_matrix()

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
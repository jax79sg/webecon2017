from ipinyouReader import ipinyouReader
import pandas as pd
import numpy as np
import time


"""
Just need to run it once!

This script does the following

1. Remove error row and save to file
    - Remove payprice > bidprice

2. onehotenc

"""

def removeErrorRows(df):
    print("Before: ", df.shape[0])
    df = df[df.bidprice >= df.payprice]
    #df = df[df.payprice >= df.slotprice] # keep this for now
    print("After: ", df.shape[0])
    return df

def savetoCSV(df, filename):
    # Save df as CSV using filename
    df.to_csv(filename, index=False,)


# ## Read in train.csv
# print("== train")
# reader = ipinyouReader("./data/train.csv")
# traindf = reader.getDataFrame()
# print("clean train")
# traindf = removeErrorRows(df)
# savetoCSV(df, "./data/train_cleaned.csv")
#
# ## Read in validate.csv
# print("== validation")
#
# reader = ipinyouReader("./data/validation.csv")
# validationdf = reader.getDataFrame()
# print("clean validation")
# validationdf = removeErrorRows(df)
# savetoCSV(df, "./data/validation_cleaned.csv")
#
# ## combine the train and val set
# combineddf = pd.concat([traindf, validationdf], axis=0)
# savetoCSV(combineddf, "./data/merged_train_validation_cleaned.csv")

## Read in combined set
print("== combined reader")
print(time.strftime("%Y-%m-%d %H:%M"))
combinedreader = ipinyouReader("./data/merged_train_validation_cleaned.csv")
print("== get train OneHotData")
print(time.strftime("%Y-%m-%d %H:%M"))
trainOneHotData,trainY = combinedreader.getOneHotData(exclude_domain=False)
# trainOneHotData,trainY = combinedreader.getOneHotData(exclude_domain=True)
# print("== save X to csv")
# print(time.strftime("%Y-%m-%d %H:%M"))
# savetoCSV(trainOneHotData, "./data/onehot_X_nodomain_merged_train_validation_cleaned.csv")
# print("== save Y to csv")
# print(time.strftime("%Y-%m-%d %H:%M"))
# savetoCSV(trainY, "./data/onehot_Y_nodomain_merged_train_validation_cleaned.csv")
print("== save Y to csv")
print(time.strftime("%Y-%m-%d %H:%M"))
savetoCSV(trainY, "./data/onehot_Y_merged_train_validation_cleaned.csv")


print("== test data reader")
print(time.strftime("%Y-%m-%d %H:%M"))
testReader = ipinyouReader("./data/test.csv")
print("== get test OneHotData")
print(time.strftime("%Y-%m-%d %H:%M"))
#testOneHotData,testY = testReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),exclude_domain=True)
# print("== save X to csv")
# print(time.strftime("%Y-%m-%d %H:%M"))
# savetoCSV(testOneHotData, "./data/onehot_X_nodomain_test.csv")
# print("== save Y to csv")
# print(time.strftime("%Y-%m-%d %H:%M"))
# savetoCSV(testY, "./data/onehot_Y_nodomain_test.csv")

testOneHotData,testY = testReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),exclude_domain=False)
print("== save X to csv")
print(time.strftime("%Y-%m-%d %H:%M"))
savetoCSV(testOneHotData, "./data/onehot_X_test.csv")
print("== save Y to csv")
print(time.strftime("%Y-%m-%d %H:%M"))
savetoCSV(testY, "./data/onehot_Y_test.csv")

print(len(trainOneHotData.columns))
print(len(testOneHotData.columns))

#
# X_train = trainOneHotData.as_matrix()
# Y_train = trainY.as_matrix()
# X_test = testOneHotData.as_matrix()
# Y_test = testY.as_matrix()
#
# print(X_train.shape)
# print(Y_train.shape)
# print(X_val.shape)
# print(Y_val.shape)




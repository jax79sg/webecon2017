from ipinyouReader import ipinyouReader
import pandas as pd
import numpy as np

"""
Just need to run it once!

This script does the following

1. Remove error row and save to file
    - Remove payprice > bidprice

2. Prune away a portion of Click=0 row

"""

def removeErrorRows(df):
    print("Before: ", df.shape[0])
    df = df[df.bidprice >= df.payprice]
    print("After: ", df.shape[0])
    return df

def prunefortrain(df, n=1):
    """
    Prune away Click=0
    :param df: Dataframe
    :param n: The number of times of click=0 has over click=0. i.e. if n=4, click=0 will be 80% of the data
    :return:
    """
    df_Click_0 = df[df.click == 0]
    df_Click_1 = df[df.click == 1]

    df_Click_0 = df_Click_0.ix[np.random.choice(df_Click_0.index, (df_Click_1.shape[0])*n)]

    print("df_Click_0: ", df_Click_0.shape[0])
    print("df_Click_1: ", df_Click_1.shape[0])

    # Concat the data vertically
    combined_set = pd.concat([df_Click_0, df_Click_1], axis=0)

    print("combined_set: ", combined_set.shape[0])
    # print(combined_set["bidid"], " ", combined_set['click'])

    combined_set = combined_set.sample(frac=1)

    # print(combined_set["bidid"], " ", combined_set['click'])

    return combined_set


def savetoCSV(df, filename):
    # Save df as CSV using filename
    df.to_csv(filename, index=False,)


## Read in train.csv
reader = ipinyouReader("../dataset/train.csv")
df = reader.getDataFrame()

df = removeErrorRows(df)
savetoCSV(df, "../dataset/train_cleaned.csv")
df = prunefortrain(df, n=4)
savetoCSV(df, "../dataset/train_cleaned_prune.csv")

## Read in validate.csv
reader = ipinyouReader("../dataset/validation.csv")
df = reader.getDataFrame()

df = removeErrorRows(df)
savetoCSV(df, "../dataset/validation_cleaned.csv")
df = prunefortrain(df, n=4)
savetoCSV(df, "../dataset/validation_cleaned_prune.csv")






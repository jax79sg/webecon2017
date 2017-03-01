## Linear model skeleton for Problem 3.
## version 0.1



"""
## Overview for problem 3 (Not sure if i understood it correctly)
1. Create a logistic regression model for CTR estimation
    y ~ x
    where y = click
    where x = a list of variables we choose
    - disregard accuracy for now (Means will not use validation data... Will fix/update in next version)

2. CTR estimation
- Estimate click for every record in the test set using the model above
    - Assume all same advertiser for now (Will fix in next version)
- Compute pCTR = sumofclicks/alltestrecords

3. Compute bid = base_bid x pCTR/avgCTR
- A bit lost here, pCTR and avgCTR ??

"""


import numpy as np
from patsy import patsy
from sklearn.linear_model import LogisticRegression
import ipinyouReader as ipinyouReader

# #List of column names. To be copy and pasted (as needed) in the formula for logistic regression
# click='click'
# weekday='weekday'
# hour='hour'
# bidid='bidid'
# logtype='logtype'
# userid='userid'
# useragent='useragent'
# IP='IP'
# region='region'
# city='city'
# adexchange='adexchange'
# domain='domain'
# url='url'
# urlid='urlid'
# slotid='slotid'
# slotwidth='slotwidth'
# slotheight='slotheight'
# slotvisibility='slotvisibility'
# slotformat='slotformat'
# slotprice='slotprice'
# creative='creative'
# bidprice='bidprice'
# payprice='payprice'
# keypage='keypage'
# advertiser='advertiser'
# usertag='usertag'



# load dataset
print("Reading dataset...")
trainDF = ipinyouReader.ipinyouReader("../dataset/train.csv").getDataFrame()

print("Setting up Y and X for logistic regression")
y, X =patsy.dmatrices('click ~ weekday + hour + region + city + adexchange + useragent',trainDF, return_type="dataframe")
print (X.columns)


# flatten y into a 1-D array
print("Flatten y into 1-D array")
y = np.ravel(y)



# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
print("Training Model...")
model = model.fit(X, y)

# check the accuracy on the training set
print("Training acccuracy: %5.3f" % model.score(X, y))




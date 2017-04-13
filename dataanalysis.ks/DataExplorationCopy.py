import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Stats for Problem 1 - KS TAN

"""

class csvReader():
    def __init__(self, filename, engine='c'):
        self._dataframe = pd.read_csv(filename, delimiter=',',  header=0, engine=engine)

    def getDataFrame(self):
        return self._dataframe

class StatsCheck():
    def __make_autopct(self,values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct

    def summary(self,datasettocheck,colNames=[]):
        for colName in colNames:
            print(datasettocheck[colName].describe())


    def noOfClick(self, datasettocheck):
        totalrecords=datasettocheck.shape[0]
        totalclicks=datasettocheck[datasettocheck['click']==1].shape[0]
        totalnonclicks = datasettocheck[datasettocheck['click'] == 0].shape[0]
        print("totalclicks: ",totalclicks, " totalnonclicks:",totalnonclicks)
        labels = 'Clicks', 'Non-clicks'
        sizes = [totalclicks, totalnonclicks]
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct=self.__make_autopct(sizes),shadow=False, startangle=90, colors=['red','cyan'])
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # titletext="No of clicks in "+(nameofdataset)
        # ax1.set_title(titletext)
        plt.show()

    def impressionByAdvertiser(self, datasettocheck):
        print("No of impressions grouped by advertiser")
        impressions=(datasettocheck.groupby(['advertiser'])['click'].count())
        print(impressions)
        return impressions

    def clicksByAdvertiser(self, datasettocheck):
        print("No of clicks grouped by advertiser")
        clicks=(datasettocheck.groupby(['advertiser'])['click'].sum())
        print(clicks)
        return clicks

    def costsByAdvertiser(self, datasettocheck):
        print("Total cost grouped by advertiser")
        costs=(datasettocheck.groupby(['advertiser'])['payprice'].sum())
        print(costs)
        return costs


    def ctrGroupby(self,dataset, advertisers=[], groupby=None):
        dataset = dataset[dataset.advertiser.isin(advertisers)]

        groupby_ad=dataset.groupby(groupby).agg({'click':'sum','payprice':'sum'})

        groupby_ad_count=dataset.groupby(groupby)['click'].count().to_frame()
        groupby_ad_count.columns=['impression']
        basic=groupby_ad.join(groupby_ad_count)


        groupby_ad_cpc=(basic['payprice']/basic['click']).to_frame()
        groupby_ad_cpc.columns=['cpc']

        groupby_ad_ctr=(basic['click']/basic['impression']).to_frame()
        groupby_ad_ctr.columns=['ctr']

        groupby_ad_cpm=(basic['payprice']/(basic['impression']/1000)).to_frame()
        groupby_ad_cpm.columns=['cpm']


        final=basic.join(groupby_ad_ctr).join(groupby_ad_cpm).join(groupby_ad_cpc)
        print(type(final))
        print(final)
        return final
        pass

    def printBasicStats(self, dataset):
        groupby_ad=dataset.groupby(['advertiser']).agg({'click':'sum','payprice':'sum'})

        groupby_ad_count=dataset.groupby(['advertiser'])['click'].count().to_frame()
        groupby_ad_count.columns=['impression']
        basic=groupby_ad.join(groupby_ad_count)


        groupby_ad_cpc=(basic['payprice']/basic['click']).to_frame()
        groupby_ad_cpc.columns=['cpc']

        groupby_ad_ctr=(basic['click']/basic['impression']).to_frame()
        groupby_ad_ctr.columns=['ctr']

        groupby_ad_cpm=(basic['payprice']/(basic['impression']/1000)).to_frame()
        groupby_ad_cpm.columns=['cpm']


        final=basic.join(groupby_ad_ctr).join(groupby_ad_cpm).join(groupby_ad_cpc)
        print(type(final))
        print(final)
        return final

    def plotPlotForAdvertiserUsage(self, dataset):
        sns.set(style="whitegrid", color_codes=True)
        dataset= dataset[dataset['advertiser'].isin([2997,3476])]
        print(list(dataset))
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        sns.countplot(x="os", data=dataset, hue='advertiser',ax=ax1);
        sns.countplot(x="browser", data=dataset, hue='advertiser',ax=ax2);
        sns.countplot(x="adexchange", data=dataset, hue='advertiser',ax=ax3);
        sns.plt.show()


    def correlationFeatures(self,dataset):
        print(list(dataset))

        dataset=dataset.filter(items=['click', 'weekday', 'hour', 'userid', 'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage', 'advertiser', 'os', 'browser'])
        corr=dataset.corr()
        print(corr)
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
        sns.plt.show()

class SanityCheck():
    def checkColForValues(self, datasetToCheck, colNameToCheck, listOfValues):
        print("Checking column ",colNameToCheck, " for values outside of ",listOfValues)
        print(datasetToCheck[~datasetToCheck[colNameToCheck].isin(listOfValues)].filter(items=[colNameToCheck]))



    def checkForNull(self,datasetToCheck, colNameToCheck):
        print("Checking ",colNameToCheck, " for null ")
        for colname in colNameToCheck:
            print(datasetToCheck[datasetToCheck[colname].astype(str) == 'null'].filter(items=[colname]).count())

    def checkForNA(self,datasetToCheck, colNameToCheck):
        print("Checking ",colNameToCheck, " for NA ")
        for colname in colNameToCheck:
            print(datasetToCheck[datasetToCheck[colname].astype(str) == 'Na'].filter(items=[colname]).count())
            print(datasetToCheck[datasetToCheck[colname].astype(str) == 'NA'].filter(items=[colname]).count())

    def checkForBidpriceLessThanPayprice(self,datasetToCheck):
        print("No of Payprice>Bidprice: ",datasetToCheck[datasetToCheck['payprice'] > datasetToCheck['bidprice']].filter(items=['payprice']).count())

class cleanup():

    def balanceValidationset(self,datasettobalance):
        click1set=datasettobalance.loc[datasettobalance['click'] == 1]
        click0set=datasettobalance.loc[datasettobalance['click'] == 0]
        click0set=click0set.reset_index(drop=True)
        click0set=click0set[:click1set.shape[0]]
        finalset=click1set.append(click0set)
        finalset = finalset.sample(frac=1).reset_index(drop=True)
        print("Resampled set: No of click=1:",finalset.loc[finalset['click'] == 1].shape[0]," No of click=0:", finalset.loc[finalset['click'] == 0].shape[0])
        return finalset



    def blankToNan(self, datasetToCheck):
        replaced=datasetToCheck.replace(r'\s+', np.nan, regex=True)
        return replaced

    def cleanup_bidpricelesspayprice(self,datasettoclean):
        print("\nCleaning up..\nOriginal size: ",datasettoclean.shape)
        cleanedup=datasettoclean[~(datasettoclean['payprice'] > datasettoclean['bidprice'])]
        cleanedupIndexReset=cleanedup.reset_index(drop=True)
        print("After clean:",cleanedupIndexReset.shape)
        return cleanedupIndexReset

    def cleanup_usertags(self, datasetToClean):
        print("\nCLEANING up user tags")
        print("Original shape:", datasetToClean.shape)
        combined_set = pd.concat([datasetToClean, datasetToClean.usertag.astype(str).str.strip('[]').str.get_dummies(',').astype(np.uint8)],axis=1)
        combined_set.rename(columns={'null': 'unknownusertag'}, inplace=True)
        print("Merged shape:", combined_set.shape)
        print("Dropping usertag")
        usertagdropped = combined_set.drop('usertag', axis=1)
        hashtagdropped = usertagdropped.drop('###############################################################################################################################################################################################################################################################', axis=1)

        print("Columns:", list(hashtagdropped))
        return hashtagdropped

    def cleanup_useragent(self, datasetToClean):
        print("\nCLEANING up user agent")
        print("Original shape:", datasetToClean.shape)
        print("\nSplitting useragent")
        useragendDf=pd.DataFrame(datasetToClean.useragent.str.split('_',1).tolist(), columns=['os','browser'])
        print("Spliited as shape:",useragendDf.shape)
        # print("Sample print:",useragendDf)

        print("Merging back into original")
        print("Original shape:",datasetToClean.shape)
        print("useragendDf shape:", useragendDf.shape)
        combineddf=pd.concat([datasetToClean,useragendDf],axis=1)
        print("Merged as one:",combineddf.shape)
        # print("Sample print:",combineddf)

        print("Dropping useragent")
        useragentdropped=combineddf.drop('useragent',axis=1)
        print("Final:",useragentdropped.shape)
        # print("Sample print:",useragentdropped)

        return useragentdropped

    def cleanup_unusedcolumns(self, datasettoclean, columnstoremove=[]):
        print("\nCleaning up unused cols")
        print("Original shape:",datasettoclean.shape)
        colsdropped = datasettoclean.drop(columnstoremove, axis=1)
        print("colsdropped shape:", colsdropped.shape)
        return colsdropped


def getBidPrice(clickProb,noBidThreshold,bidRange,minBid,sigmoiddegree):
    """
    :param clickProb: Single click prob (Not an array)
    :return:
    """
    bid=-1
    if (clickProb>noBidThreshold):
        bid = (sigmoid(clickProb,noBidThreshold,sigmoiddegree)) * bidRange + minBid
    return bid

def sigmoid(x, threshold=None,sigmoiddegree=-30):
    import numpy as np
    sigmoidthreshold=-0.2-threshold
    a=1/(1+np.exp(sigmoiddegree*(x+sigmoidthreshold)))
    return a

def plotSigmoid():
    import matplotlib.pyplot as plt
    import numpy as np
    Xaxis = np.arange(0., 1., 0.001)
    Yaxis = np.linspace(0, len(Xaxis), len(Xaxis))

    minBid=220
    c=minBid
    priceRange=100
    f = np.vectorize(getBidPrice)

    # Yaxis = (sigmoid(Xaxis))*priceRange+minBid
    Yaxis=f(Xaxis, noBidThreshold=0.4,bidRange=200,minBid=220,sigmoiddegree=-30)
    plt.plot(Xaxis, Yaxis)
    plt.show()

plotSigmoid()   #Setting up Threshold-Sigmoid bid pricing
sc=SanityCheck()
statsc=StatsCheck()
c=cleanup()

# ### Sanity checks. ####################################
# print("-------- Size stats ---------")
# print("Train row/col:",trainDF.shape)
# print("Validation row/col:",validationDF.shape)
# print("Test row/col:",testDF.shape)
#
# print("-------- Sanity check ---------")
# print("\n==========Sanity check Scanning validation")
# sc.checkColForValues(validationDF,'click',[0,1])
# sc.checkColForValues(validationDF,'weekday',list(range(0,6+1)))
# sc.checkColForValues(validationDF,'hour',list(range(0,23+1)))
# sc.checkColForValues(validationDF,'logtype',[1])
# sc.checkForNull(validationDF,['bidid','userid','useragent','IP','region','city','adexchange','domain','slotid','slotwidth','slotheight','slotvisibility','slotformat','slotprice','creative','bidprice','payprice','keypage','advertiser','usertag'])
# sc.checkForNA(validationDF,['bidid','userid','useragent','IP','region','city','adexchange','domain','slotid','slotwidth','slotheight','slotvisibility','slotformat','slotprice','creative','bidprice','payprice','keypage','advertiser','usertag'])
# sc.checkForBidpriceLessThanPayprice(validationDF)
#
# print("\n==========Sanity check Scanning train")
# sc.checkColForValues(trainDF,'click',[0,1])
# sc.checkColForValues(trainDF,'weekday',list(range(0,6+1)))
# sc.checkColForValues(trainDF,'hour',list(range(0,23+1)))
# sc.checkColForValues(trainDF,'logtype',[1])
# sc.checkForNull(trainDF,['bidid','userid','useragent','IP','region','city','adexchange','domain','slotid','slotwidth','slotheight','slotvisibility','slotformat','slotprice','creative','bidprice','payprice','keypage','advertiser','usertag'])
# sc.checkForNA(trainDF,['bidid','userid','useragent','IP','region','city','adexchange','domain','slotid','slotwidth','slotheight','slotvisibility','slotformat','slotprice','creative','bidprice','payprice','keypage','advertiser','usertag'])
# sc.checkForBidpriceLessThanPayprice(trainDF)
#
# print("\n==========Sanity check Scanning test")
# sc.checkColForValues(testDF,'weekday',list(range(0,6+1)))
# sc.checkColForValues(testDF,'hour',list(range(0,23+1)))
# sc.checkColForValues(testDF,'logtype',[1])
# sc.checkForNull(testDF,['bidid','userid','useragent','IP','region','city','adexchange','domain','slotid','slotwidth','slotheight','slotvisibility','slotformat','slotprice','creative','keypage','advertiser','usertag'])
# sc.checkForNA(testDF,['bidid','userid','useragent','IP','region','city','adexchange','domain','slotid','slotwidth','slotheight','slotvisibility','slotformat','slotprice','creative','keypage','advertiser','usertag'])




# ### Preprocessing: Cleaning up, steps 1 to 6. ####################################
# ### Step 1: Removing invalid records
# print("Reading raw training set")
# trainDF=csvReader("../data.original/train.csv").getDataFrame()
# print("Reading raw validation set")
# validationDF=csvReader("../data.original/validation.csv").getDataFrame()
# # print("Reading raw test set")
# # testDF=csvReader("../data.original/test.csv").getDataFrame()
#
# sizeofTrain=trainDF.shape[0]
# sizeofValidate=validationDF.shape[0]
# print("trainDF shape:",trainDF.shape)
# print("validationDF shape:",validationDF.shape)
#
#
# print("-------- Removing invalid train---------")
# trainDFremoved=c.cleanup_bidpricelesspayprice(trainDF)
# print("-------- Removing invalid validation---------")
# validationDFremoved=c.cleanup_bidpricelesspayprice(validationDF)
#
# print("\nSaving both train and validation")
# trainDFremoved.to_csv("step1_trainDFremoved.csv")
# validationDFremoved.to_csv("step1_validationDFremoved.csv")
# ### End of Step 1



# ### Step 2: Merging validate and train
# print("Reading  training set")
# trainDF=csvReader("step1_trainDFremoved.csv").getDataFrame()
# print("Reading  validation set")
# validationDF=csvReader("step1_validationDFremoved.csv").getDataFrame()
#
# sizeofTrain=trainDF.shape[0]
# sizeofValidate=validationDF.shape[0]
# print("trainDF shape:",trainDF.shape)
# print("validationDF shape:",validationDF.shape)
#
# combinedDF=pd.concat([trainDF,validationDF])
# combinedIndexResetDF=combinedDF.reset_index(drop=True)
# print("combinedIndexResetDF shape:",combinedIndexResetDF.shape)
# combinedIndexResetDF.to_csv("step2_combinedIndexResetDF.csv")
# assert(combinedIndexResetDF.shape[0]==trainDF.shape[0]+validationDF.shape[0])
# assert(combinedIndexResetDF.shape[1]==trainDF.shape[1]==validationDF.shape[1])
# ### End of Step 2


# ### Step 3: Replace blanks with NaN
# print("Reading  combined set")
# combinedDF=csvReader("step2_combinedIndexResetDF.csv").getDataFrame()
# print("combinedDF shape:",combinedDF.shape)
#
# combinedDFblankedNan=c.blankToNan(combinedDF)
# print("combinedDFblankedNan shape:",combinedDFblankedNan.shape)
#
# combinedDFblankedNan.to_csv("step3_combinedDFblankedNan.csv")
# ### End of Step 3


# ### Step 4: Clean up useragents
# print("Reading  step3_combinedDFblankedNan set")
# combinedDF=csvReader("step3_combinedDFblankedNan.csv").getDataFrame()
# print("combinedDF shape:",combinedDF.shape)
#
# cleanedCombinedDFuseragent=c.cleanup_useragent(combinedDF)
# print("cleanedCombinedDFuseragent shape:",cleanedCombinedDFuseragent.shape)
# cleanedCombinedDFuseragent.to_csv("step4_cleanedCombinedDFuseragent.csv")
# ### End of Step 4



# ### Step 5: Clean up usertags
# print("Reading  step4_cleanedCombinedDFuseragent set")
# combinedDF=csvReader("step4_cleanedCombinedDFuseragent.csv").getDataFrame()
# print("combinedDF shape:",combinedDF.shape)
#
# cleanedCombinedDFusertags=c.cleanup_usertags(combinedDF)
# print("cleanedCombinedDFusertags shape:",cleanedCombinedDFusertags.shape)
# cleanedCombinedDFusertags.to_csv("step5_cleanedCombinedDFusertags.csv")
# ### End of Step 5

# ### Step 5: Clean up unused cols
# print("Reading  step5_cleanedCombinedDFusertags set")
# combinedDF=csvReader("step5_cleanedCombinedDFusertags.csv", engine='c').getDataFrame() #c engine will have memory issues for such large files
# print("combinedDF shape:",combinedDF.shape)
# print("combinedDF shape:",list(combinedDF))
# cleanedCombinedDFusertags=c.cleanup_unusedcolumns(combinedDF,['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'])
# print("cleanedCombinedDFusertags shape:",list(cleanedCombinedDFusertags))
# print("cleanedCombinedDFusertags shape:",cleanedCombinedDFusertags.shape)
# cleanedCombinedDFusertags.to_csv("step5_cleanedCombinedDFusertags.csv")
# ### End of Step 5


# ### Step 6: Split back
# print("Reading  training set")
# trainDF=csvReader("step1_trainDFremoved.csv").getDataFrame()
# print("trainDF shape:", trainDF.shape)
# print("Reading  validation set")
# validationDF=csvReader("step1_validationDFremoved.csv").getDataFrame()
# print("validationDF shape:", validationDF.shape)
#
# sizeofTrain=trainDF.shape[0]
# sizeofValidate=validationDF.shape[0]
#
#
# print("Reading  step5_cleanedCombinedDFusertags set")
# combinedDF=csvReader("step5_cleanedCombinedDFusertags.csv").getDataFrame()
# print("combinedDF shape:",combinedDF.shape)
#
# print(" Splitting into training and validation  ")
# trainCleanDf=combinedDF.iloc[0:sizeofTrain]
# trainCleanIndexedDf=trainCleanDf.reset_index(drop=True)
# validationCleanDf=combinedDF.iloc[sizeofTrain-1:-1]
# validateCleanIndexedDf=validationCleanDf.reset_index(drop=True)
# print("trainCleanDf shape:",trainCleanIndexedDf.shape)
# print("trainCleanDf col:",list(trainCleanIndexedDf))
# print("validationCleanDf shape:",validateCleanIndexedDf.shape)
# print("validateCleanIndexedDf col:",list(validateCleanIndexedDf))
# trainCleanIndexedDf.to_csv("step6_trainCleanIndexedDf.csv")
# validateCleanIndexedDf.to_csv("step6_validateCleanIndexedDf.csv")
#
# ### End of Step 6




#
#
# print("-------- Statistics check ---------")
# print("Reading  training set")
# trainDF=csvReader("step6_trainCleanIndexedDf.csv").getDataFrame()
# print("trainDF shape:", trainDF.shape)

print("Reading validation set")
validateDF=csvReader("../data.final/validation_cleaned.csv").getDataFrame()
print("validateDF shape:", validateDF.shape)

# print("Reading  val set")
# valDF=csvReader("step6_validateCleanIndexedDf.csv").getDataFrame()
# print("valDF shape:", valDF.shape)

# statsc.noOfClick(trainDF)
# clicks=statsc.clicksByAdvertiser(trainDF)
# impressions=statsc.impressionByAdvertiser(trainDF)
# costs=statsc.costsByAdvertiser(trainDF)

# stats=statsc.printBasicStats(valDF)


# statsc.plotPlotForAdvertiserUsage(trainDF)
# statsc.correlationFeatures(trainDF)
# statsc.summary(trainDF,['bidprice','payprice','slotprice'])

# validateDF=cleanup().balanceValidationset(validateDF)
# validateDF.to_csv("../data.pruned/validation_cleaned_balanced.csv")





# cleanedCombinedDFusertags=c.cleanup_usertags(clceanedCombinedDFuseragent)
# cleanedCombinedDFusertags.tocsv("combinedCleaned.csv")
# print(" -------- Splitting into training and validation -------- ")
# trainCleanDf=cleanedCombinedDFusertags.iloc[0:sizeofTrain+1]
# trainCleanIndexedDf=trainCleanDf.reset_index()
# validationCleanDf=cleanedCombinedDFusertags.iloc[sizeofTrain+2:-1]
# validateCleanIndexedDf=validationCleanDf.reset_index()
# print("trainCleanDf shape:",trainCleanIndexedDf.shape)
# print("validationCleanDf shape:",validateCleanIndexedDf.shape)
#
# print(" -------- Saving to files -------- ")
# trainCleanIndexedDf.tocsv("trainCleanIndexedDf.csv")
# validateCleanIndexedDf.tocsv("validateCleanIndexedDf.csv")
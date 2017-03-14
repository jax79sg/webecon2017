import pandas as pd
import numpy as np
import re

class ipinyouReader():
    def __init__(self, filename, header=0):
        self.__dataframe = pd.read_csv(filename, delimiter=',', low_memory=False, header=header)

    def getDataFrame(self):
        return self.__dataframe

    def getTrainData(self):
        """ Return the all data in numpy array format. """
        return self.__dataframe.as_matrix()

    def getTestData(self):
        """ Return data in test format. i.e. without click, bidprice and payprice """
        return self.__dataframe.as_matrix(['weekday',   #0
                                           'hour',      #1
                                           'bidid',     #2
                                           'logtype',   #3
                                           'userid',    #4
                                           'useragent', #5
                                           'IP',        #6
                                           'region',    #7
                                           'city',      #8
                                           'adexchange',#9
                                           'domain',    #10
                                           'url',       #11
                                           'urlid',     #12
                                           'slotid',    #13
                                           'slotwidth', #14
                                           'slotheight',#15
                                           'slotvisibility',#16
                                           'slotformat',#17
                                           'slotprice', #18
                                           'creative',  #19
                                           'keypage',   #20
                                           'advertiser',#21
                                           'usertag'])  #22

    def getOneHotDatav1(self, train_cols=[]):
        """ Return categorical data in one-hot encoding format, remove other columns."""

        ### Create new data frame with columns split for combined columns
        source_df = self.__dataframe.copy()
        # useragent split
        source_df['user_platform'] = source_df.useragent.str.split('_').str.get(0)
        source_df['user_browser'] = source_df.useragent.str.split('_').str.get(1)
        # ip split
        source_df['ip_block'] = source_df.IP.str.split('.').str.get(0)

        ### one hot encoding for relevant columns
        # simple columns
        onehot_df = pd.get_dummies(source_df, columns=['weekday', 'hour',  # ])
                                                       'user_platform', 'user_browser', 'ip_block',
                                                       'region', 'city', 'adexchange', 'domain',
                                                       'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
                                                       'creative', 'keypage', 'advertiser',
                                                       ])

        # usertags
        onehot_df = onehot_df.join(source_df.usertag.astype(str).str.strip('[]').str.get_dummies(',').astype(np.uint8))

        ### Drop these non-categorical data
        onehot_df.drop(['click', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'url',
                        'urlid', 'slotid', 'slotprice', 'bidprice', 'payprice', 'usertag'], axis=1, inplace=True)

        ### get the y values too
        y_values = source_df[['click', 'bidprice', 'payprice']]

        ### if we are using an existing column def (i.e we are processing test/validation data)
        if len(train_cols) > 0:
            new_onehot_df = pd.DataFrame(data=onehot_df, columns=train_cols)
            new_onehot_df.fillna(0, inplace=True)  # Fill any NaNs
            return new_onehot_df, y_values
        else:
            return onehot_df, y_values

    def getOneHotData(self,train_cols=[],exclude_domain=True,domain_keep_prob=0.05):
        """ Return categorical data in one-hot encoding format, remove other columns."""

        ### Create new data frame with columns split for combined columns
        source_df = self.__dataframe.copy()
        ## useragent split
        source_df['user_platform'] = source_df.useragent.str.split('_').str.get(0)
        source_df['user_browser'] = source_df.useragent.str.split('_').str.get(1)
        ## ip split
        source_df['ip_block'] = source_df.IP.str.split('.').str.get(0)

        ## domain narrowing skewed data
        if exclude_domain == False:
            # if we are using an existing column def (i.e we are processing test/validation data)
            if len(train_cols) > 0:
                domain_regex = re.compile('^domain_')
                train_domains=[domain.split('_')[1] for domain in train_cols if domain_regex.search(domain)]
                mask_train_domains=source_df['domain'].isin(train_domains)
                source_df['domain'].where(pd.isnull(source_df['domain'].mask(mask_train_domains)) == True, "othertoofew", inplace=True)
            else:
                domain_counts=source_df['domain'].value_counts()
                cutoff_threshold=int(domain_keep_prob * len(domain_counts))
                if domain_keep_prob == 1.0:
                    print('Keep (all) {} number of unique domains'.format(cutoff_threshold))
                else:
                    print('Keep {} number of unique domains'.format(cutoff_threshold))
                    count_threshold=domain_counts[cutoff_threshold]
                    print('Cut domain data at less than {} frequency'.format(count_threshold))
                    #this gives a df of same shape where the entry is value_count -- source_df['domain'].map(source_df['domain'].value_counts())
                    source_df['domain'].where(source_df['domain'].map(domain_counts) > count_threshold, "othertoofew",inplace=True) #will end up with > 100 being left ~1633/24972 on full DS

            ### one hot encoding for relevant columns
            ## simple columns
            onehot_df = pd.get_dummies(source_df, columns=['weekday', 'hour',  # ])
                                                           'user_platform', 'user_browser', 'ip_block',
                                                           'region', 'city', 'adexchange','domain',
                                                           'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
                                                           'creative', 'keypage', 'advertiser',
                                                           ])
            #onehot_df = pd.get_dummies(onehot_df, prefix='domain',columns=['domain', ]) # need to explicitly prefix
            #onehot_df = pd.get_dummies(onehot_df, columns=['domain', ]).astype(np.uint16) # can't use this directly with get_dummies with full dataset, too high cardinality (24972) OOM in pandas
        else: #leave out domain
            ### one hot encoding for relevant columns
            ## simple columns
            onehot_df = pd.get_dummies(source_df, columns=['weekday', 'hour',  # ])
                                                           'user_platform', 'user_browser', 'ip_block',
                                                           'region', 'city', 'adexchange',
                                                           'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
                                                           'creative', 'keypage', 'advertiser',
                                                           ])
        # usertags
        onehot_df = onehot_df.join(source_df.usertag.astype(str).str.strip('[]').str.get_dummies(','))#.astype(np.uint16))

        #Drop these non-categorical data
        drop_cols=['bidid', 'logtype', 'userid', 'useragent', 'IP', 'url',
                            'urlid', 'slotid', 'slotprice', 'usertag',
                            ]
        y_cols=['bidid']
        if 'click' in source_df.columns: # i.e there is click data in hte dataset
            ### add this to drop cols
            drop_cols+=['click','bidprice','payprice']
            y_cols+=['click','bidprice','payprice']

        if exclude_domain:
            drop_cols.append('domain')

        ### Drop these non-categorical data
        onehot_df.drop(drop_cols, axis=1, inplace=True)
        ### take these as y
        y_values = source_df[y_cols]

        ### if we are using an existing column def (i.e we are processing test/validation data)
        if len(train_cols) > 0:
            new_onehot_df = pd.DataFrame(data=onehot_df, columns=train_cols)
            new_onehot_df.fillna(0,inplace=True) #Fill any NaNs
            return new_onehot_df, y_values
        else:
            return onehot_df,y_values

    def getResult(self):
        """ Return data in result format for evaluation. i.e. bidid, bidprice """
        return self.__dataframe.as_matrix()



class ipinyouReaderWithEncoding():
    def getTrainValidationTestDD(self, trainFilename, validationFilename, testFilename, header=0):
        """
        1. Reads train, validation and test file
        2. Represent non-numeric column with a unique int for each category/unique string
        3. Returns trainDF, validationDF, testDF, lookupDict

        :param trainFilename: Training Filename
        :param validationFilename: Validation Filename
        :param testFilename:  Test Filename
        :param header: csv has header? 0 or None
        :return: trainDF, validationDF, testDF, lookupDict
        """
        print("Reading Train: ", trainFilename)
        traindf = pd.read_csv(trainFilename, delimiter=',', low_memory=False, header=header)

        print("Reading Validate: ", validationFilename)
        validationdf = pd.read_csv(validationFilename, delimiter=',', low_memory=False, header=header)

        print("Reading Test: ", testFilename)
        testdf = pd.read_csv(testFilename, delimiter=',', low_memory=False, header=header)

        # Concat the data vertically
        combined_set = pd.concat([traindf, validationdf, testdf], axis=0)
        # print(combined_set.info())
        dict = {}
        # Loop through all columns in the dataframe
        print("Encoding all features in columns")
        for feature in combined_set.columns:

            # Only apply for columns with categorical strings
            if combined_set[feature].dtype == 'object':

                original = combined_set[feature]
                # Replace strings with an integer
                combined_set[feature] = pd.Categorical(combined_set[feature]).codes

                replaced = combined_set[feature]

                # TODO: Need to find a way to speed this up
                if feature == 'bidid':
                    colDict = {}
                    for i in range(len(original)):
                        # print("ttt: ", original.iloc[i], "    ", replaced.iloc[i])
                        if replaced.iloc[i] not in colDict:
                            colDict[replaced.iloc[i]] = original.iloc[i]
                    dict[feature] = colDict

        train = combined_set[:traindf.shape[0]]
        validation = combined_set[traindf.shape[0]:(traindf.shape[0]+validationdf.shape[0])]
        test = combined_set[(traindf.shape[0]+validationdf.shape[0]):]

        print("Length of Train: ", train.shape[0])
        print("Length of Validation: ", validation.shape[0])
        print("Length of Test: ", test.shape[0])

        return train, validation, test, dict

        # print("dict", dict)

    def getTrainValidationTestDF_V2(self, trainFilename, validationFilename, testFilename, header=0):
        """
        Added data cleansing
        - adexchange == null changed to value 0
        - slotformat == NA changed to value 2
        - Dropped logtype
        - Dropped urlid
        - Added column 'mobileos'
        - Added column 'slotdimension'


        1. Reads train, validation and test file
        2. Represent non-numeric column with a unique int for each category/unique string
        3. Returns trainDF, validationDF, testDF, lookupDict

        :param trainFilename: Training Filename
        :param validationFilename: Validation Filename
        :param testFilename:  Test Filename
        :param header: csv has header? 0 or None
        :return: trainDF, validationDF, testDF, lookupDict
        """
        print("Reading Train: ", trainFilename)
        traindf = pd.read_csv(trainFilename, delimiter=',', low_memory=False, header=header)

        print("Reading Validate: ", validationFilename)
        validationdf = pd.read_csv(validationFilename, delimiter=',', low_memory=False, header=header)

        print("Reading Test: ", testFilename)
        testdf = pd.read_csv(testFilename, delimiter=',', low_memory=False, header=header)

        # Concat the data vertically
        combined_set = pd.concat([traindf, validationdf, testdf], axis=0)

        # Change adexchange null to 0
        combined_set.loc[combined_set['adexchange'] == 'null', 'adexchange'] = 0
        combined_set.adexchange = combined_set.adexchange.astype(int)

        # Change slotformat Na to 2
        combined_set.loc[combined_set['slotformat'] == 'Na', 'slotformat'] = 2
        combined_set.slotformat = combined_set.slotformat.astype(int)

        combined_set['mobileos'] = np.where(((combined_set['useragent'] == 'android_safari') |
                                   (combined_set['useragent'] == 'android_other') |
                                   (combined_set['useragent'] == 'ios_safari') |
                                   (combined_set['useragent'] == 'android_chrome') |
                                   (combined_set['useragent'] == 'android_opera') |
                                   (combined_set['useragent'] == 'android_maxthon') |
                                   (combined_set['useragent'] == 'ios_other') |
                                   (combined_set['useragent'] == 'android_firefox') |
                                   (combined_set['useragent'] == 'android_sogou') |
                                   (combined_set['useragent'] == 'android_ie')
                                   ), 1, 0)

        combined_set['slotdimension'] = combined_set['slotwidth'].map(str) + "x" + combined_set['slotheight'].map(str)

        combined_set = pd.concat([combined_set, combined_set.usertag.astype(str).str.strip('[]').str.get_dummies(',').astype(np.uint8)], axis=1)
        combined_set.rename(columns={'null': 'unknownusertag'}, inplace=True)

        # Appended X to all column name with digit only for patsy
        updatedName = {}
        for i in list(combined_set):
            if i.isdigit():
                updatedName[i] = 'X' + i

        combined_set.rename(columns=updatedName, inplace=True)

        combined_set['os'] = combined_set.useragent.str.split('_').str.get(0)
        combined_set['browser'] = combined_set.useragent.str.split('_').str.get(1)
        combined_set['ip_block'] = combined_set.IP.str.split('.').str.get(0) #+"."+combined_set.IP.str.split('.').str.get(1)

        # Add Frequency Feature
        def createFreqColumn(df, columnName):
            freq = pd.DataFrame(df[columnName].value_counts())
            freq.rename(columns={columnName: columnName+'_freq'}, inplace=True)
            freq.index.name = columnName
            freq.reset_index(inplace=True)
            return pd.merge(df, freq, how='left', on=columnName)

        combined_set = createFreqColumn(combined_set, 'region')
        combined_set = createFreqColumn(combined_set, 'city')
        combined_set = createFreqColumn(combined_set, 'ip_block')
        combined_set = createFreqColumn(combined_set, 'adexchange')
        combined_set = createFreqColumn(combined_set, 'os')
        combined_set = createFreqColumn(combined_set, 'browser')
        combined_set = createFreqColumn(combined_set, 'mobileos')
        combined_set = createFreqColumn(combined_set, 'slotformat')
        combined_set = createFreqColumn(combined_set, 'slotdimension')
        combined_set = createFreqColumn(combined_set, 'slotvisibility')
        combined_set = createFreqColumn(combined_set, 'slotwidth')
        combined_set = createFreqColumn(combined_set, 'slotheight')
        combined_set = createFreqColumn(combined_set, 'weekday')
        combined_set = createFreqColumn(combined_set, 'hour')

        # Add CTR Feature


        # combined_set.ix[combined_set.slotprice.between(0, 20), 'slotpricebucket'] = 1
        # combined_set.ix[combined_set.slotprice.between(21, 40), 'slotpricebucket'] = 2
        # combined_set.ix[combined_set.slotprice.between(41, 60), 'slotpricebucket'] = 3
        # combined_set.ix[combined_set.slotprice.between(61, 80), 'slotpricebucket'] = 4
        # combined_set.ix[combined_set.slotprice.between(81, 100), 'slotpricebucket'] = 5
        # combined_set.ix[combined_set.slotprice.between(101, 120), 'slotpricebucket'] = 6
        # combined_set.ix[combined_set.slotprice.between(121, 140), 'slotpricebucket'] = 7
        # combined_set.ix[combined_set.slotprice.between(141, 160), 'slotpricebucket'] = 8
        # combined_set.ix[combined_set.slotprice.between(161, 180), 'slotpricebucket'] = 9
        # combined_set.ix[combined_set.slotprice.between(181, 5000), 'slotpricebucket'] = 10
        # combined_set['slotpricebucket'] = combined_set['slotpricebucket'].astype(np.uint8)


        # Useless column that contains only 1 unique value
        # Remove them to save some memory
        combined_set.pop('logtype')
        combined_set.pop('urlid')
        combined_set.pop('usertag')

        # print(combined_set.info())

        # Loop through all columns in the dataframe
        for feature in combined_set.columns:

            # Only apply for columns with categorical strings
            # if combined_set[feature].dtype == 'object':
            if feature == 'userid' or \
                feature == 'useragent' or \
                feature == 'IP' or \
                feature == 'domain' or \
                feature == 'url' or \
                feature == 'slotid' or \
                feature == 'slotvisibility' or \
                feature == 'creative' or \
                feature == 'keypage' or \
                feature == 'slotdimension' or \
                feature == 'os' or \
                feature == 'browser' or \
                feature == 'ip_block':

                original = combined_set[feature]
                # Replace strings with an integer
                combined_set[feature] = pd.Categorical(combined_set[feature]).codes

        # print(combined_set.info())

        train = combined_set[:traindf.shape[0]]
        validation = combined_set[traindf.shape[0]:(traindf.shape[0]+validationdf.shape[0])]
        test = combined_set[(traindf.shape[0]+validationdf.shape[0]):]

        print("Length of Train: ", train.shape[0])
        print("Length of Validation: ", validation.shape[0])
        print("Length of Test: ", test.shape[0])

        return train, validation, test

        # print("dict", dict)

# reader_encoded = ipinyouReaderWithEncoding()
# reader_encoded.getTrainValidationTestDD("../dataset/debug.csv", "../dataset/debug.csv", "../dataset/test.csv")


# reader = ipinyouReader("debug.csv")
# print ("A")
# debug = reader.getDataFrame()
# print ("B")
# test = reader.getMatrix()
# print("C")
#
# print(test.shape)
# print (test[0])
# print(debug.iloc[0])
# print(debug.iloc[0]['bidid'])

# for index, row in debug.iterrows():
#     # print(index)
#     print("row: ", row['bidid'])

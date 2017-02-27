import pandas as pd

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


    def getResult(self):
        """ Return data in result format for evaluation. i.e. bidid, bidprice """
        return self.__dataframe.as_matrix()


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

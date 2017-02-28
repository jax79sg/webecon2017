import pandas as pd


class TableReader():
    """
    Didn't use ipinyouReader because that has delimiter coded in.
    """
    def __init__(self, filename, header=0, delimiter=','):
        self.__dataframe = pd.read_csv(filename, delimiter=delimiter,low_memory=False, header=header)

    def getDataFrame(self):
        return self.__dataframe



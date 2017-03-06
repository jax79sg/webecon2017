import pandas as pd

class ResultWriter():
    def writeResult(self, filename, data):
        pd.DataFrame(data).to_csv(filename, index=False, header=True)
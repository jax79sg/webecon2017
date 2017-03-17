
from fastFM import sgd
import numpy as np

threshold=0.5

class SGDFMClassification(sgd.FMClassification):
    def predict_proba(self, X_test):
        probOfClickOne=super(SGDFMClassification, self).predict_proba(X_test)

        resultingProb = []
        for item in probOfClickOne:
            click1prob = item
            click0prob = 1 - item
            resultingProb.append([click0prob, click1prob])
        predictedProb = np.array(resultingProb)
        # print("predictedProb sample:", predictedProb.shape, predictedProb[0])
        return predictedProb


    def predict(self, X_test):
        probOfClickOne=super(SGDFMClassification, self).predict_proba(X_test)

        counter=0
        resultingChoice = []
        for item in probOfClickOne:
            if (item >= threshold):
                counter=counter+1
                resultingChoice.append(1)
            else:
                resultingChoice.append(-1)
        predicted = np.array(resultingChoice)
        print("No of ones predicted:", counter)
        return predicted

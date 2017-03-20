import ipinyouReader
from collections import defaultdict
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns #because seaborn has more beautiful plots
from sklearn.metrics import roc_curve, auc, mean_squared_error #roc_auc_score as AUC
import itertools

class Evaluator():

    def computePerformanceMetricsDF(self, budget, ourBidsDF, goldlabelsDF,verbose=False):

        self.resultDict = defaultdict(float)

        start_time = time.time()
        bidids_match=np.all(np.equal(ourBidsDF.bidid, goldlabelsDF['bidid'])) #safety check bidids match
        if bidids_match:
            #if bid price is greater than (or eq) gold's payprice - TA said it's greater or eq to payprice = win
            wonbid=np.greater_equal(ourBidsDF['bidprice'].astype(int), goldlabelsDF['payprice'])
            wonbidIndex = np.where(wonbid)
            spend = goldlabelsDF['payprice'].iloc[wonbidIndex]
            clicks = goldlabelsDF.click.iloc[wonbidIndex]

            # work backwards until we are within budget
            i = -1
            overspend = sum(spend) - budget
            while overspend > 0 and len(spend) + i > 0:
                if verbose:
                    print("{},{},{}".format(i, sum(spend), budget))
                overspend += -spend.iloc[i]
                spend.iloc[i] = 0
                clicks.iloc[i] = 0
                wonbid.iloc[i] = False
                i += -1


            self.resultDict={'won':sum(wonbid),'click':sum(clicks),'spend':sum(spend),'trimmed_bids':-i-1}
            if self.resultDict['won'] != 0:
                self.resultDict['CTR']=self.resultDict['click'] / self.resultDict['won']
                self.resultDict['CPM']=self.resultDict['spend'] / (self.resultDict['won'] / 1000)
            else:
                self.resultDict['CTR']=None
                self.resultDict['CPM']=None

            if self.resultDict['click'] != 0:
                self.resultDict['CPC'] = self.resultDict['spend'] / self.resultDict['click']
            else:
                self.resultDict['CPC'] = None

        else:
            print("Bid Ids did not match in arrays")

        print('Metrics compute time: {} seconds'.format(round(time.time() - start_time, 2)))
        return self.resultDict

    def printResult(self):
        print("Trimmed Bids:",self.resultDict['trimmed_bids'])
        print("Won: ", self.resultDict['won'])
        print("Click: ", self.resultDict['click'])
        if self.resultDict['won'] != 0:
            print("CTR: ", self.resultDict['click'] / self.resultDict['won'])
            print("CPM: ", self.resultDict['spend']/(self.resultDict['won']/1000))
        else:
            print("CTR and CPM not computed as no. of won is 0")
        print("Spend: ", self.resultDict['spend'])
        if self.resultDict['click'] != 0:
            print("Average CPC: ", self.resultDict['spend'] / self.resultDict['click'])
        else:
            print("Average CPC not computed as click is 0")



# BUDGET = 25000
#
# bidReader = ipinyouReader.ipinyouReader("result.csv", header=None)
# validationFileReader = ipinyouReader.ipinyouReader("../dataset/validation.csv")
#
# ourBids = bidReader.getResult()
# goldlabel = validationFileReader.getTrainData()
#
# myEvaluator = Evaluator(BUDGET, ourBids, goldlabel)
# myEvaluator.computePerformanceMetrics()
# myEvaluator.printResult()

class ClickEvaluator():
    def printClickPredictionScore(self, y_Pred, y_Gold):
        '''
        Compute the Precision, Recall and F1 score:

        :param y_Pred: Predicted Result
        :param y_Gold: Gold Label
        :return: precision=pCTR, recall, f1==> for click=1
        '''

        print("Number of 1 in pred: ", np.count_nonzero(y_Pred))
        # y_Pred = np.array(y_Pred)
        # print(y_Pred.shape)
        # print(y_Gold.shape)
        # clicked = np.count_nonzero(np.logical_and(y_Pred == 1, y_Gold == 1))
        # print("Number of 1 when pred==Gold: ", clicked)
        print("Number of 0 in pred: ", len(y_Pred) - np.count_nonzero(y_Pred))

        print("Number of 1 in gold: ", np.count_nonzero(y_Gold))
        print("Number of 0 in gold: ", len(y_Gold) - np.count_nonzero(y_Gold))


        # print("pCTR = (clicked / 1 in pred): ", clicked/np.count_nonzero(y_Pred))
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_Gold, y_Pred)

        for i in range(len(p)):
            print("Click=%d \tPrecision\pCTR: %5.3f \tRecall: %5.3f \tF1: %5.3f" % (i, p[i], r[i], f1[i]))

        # print("Focus on click=1 recall score and Number of 1 in pred.")

        return p[i], r[i], f1[i]#, (clicked/np.count_nonzero(y_Pred))

    def clickProbHistogram(self, pred_prob,color='g',title='Predicted probabilities',imgpath='',showGraph=False):
        plt.figure()
        # the histogram of the data
        n, bins, patches = plt.hist(pred_prob, 100, facecolor=color, alpha=0.75)
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title(title)
        if imgpath:
            plt.savefig(imgpath)

        if showGraph:
            plt.show()
        return n,bins,patches

    # Plot data https://vkolachalama.blogspot.co.uk/2016/05/keras-implementation-of-mlp-neural.html
    def clickROC(self,y_true, y_pred_prob,imgpath='',showGraph=False):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        if imgpath or showGraph:
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve (AUC={0:.4f})'.format(roc_auc))

        if imgpath:
            plt.savefig(imgpath)

        if showGraph:
            plt.show()

        print('AUC: %f' % roc_auc)
        return roc_auc

    def printRMSE(self, y_Pred, y_Gold):
        mse = mean_squared_error(y_Gold, y_Pred)

        rmse = mse ** 0.5

        print("RMSE: %.5f" %rmse)

    def compute_avgCTR(self,clicks_dataset):
        # True Train Click = 1 = 1986
        # True Val Click = 1 = 220
        # True Train CTR = 0.0007454510034874044
        # True Val CTR = 0.0007430976362739734
        #clicks_dataset is a np array with 1 if click==1 or 0 if click==0
        avgCTR = clicks_dataset.sum()/len(clicks_dataset)
        return avgCTR

    def plot_confusion_matrix(self,cm, classes, normalize=False,title='Confusion matrix',imgpath='',cmap=plt.cm.Blues, plotgraph=False, printStats=True):
        """
        Adapted from
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        :param cm: output from scikit learn's confusion_matrix
        :param classes: a list of classes/labels/targets
        :param normalize:
        :param title:
        :param cmap: Color gradients
        :param plotgraph: Displays the Confusion Matrix plot
        :param printStats: Print TP,TN,FP,FN,Accuracy,Misclassification Rate,Sensitivity/Recall/True Positive Rate, Specificity, Precision, F1,
        :return:
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if(printStats):

            TN,FP,FN,TP=cm.ravel()
            total=TN+FN+FP+TP
            actualYes=FN+TP
            actualNo=FP+TN
            accuracy=(TN+TP)/total
            misclassificationRate=(FP+FN)/total
            recall=TP/actualYes
            precision=TP/(TP+FP)
            f1=2 * (precision * recall) / (precision + recall)
            print("-----> Confusion Matrix stats from Postive aspect <-----")
            print("TN:",TN)
            print("FN:", FN)
            print("FP:", FP)
            print("TP:", TP)
            print("Precision:",precision)
            print("Recall:",recall)
            print("F1:",f1)
            print("Accuracy:", accuracy)
            print("Misclassification Rate:",misclassificationRate)
            print("Baselines: total:",total," Actual Click:",actualYes, " Actual No-click:", actualNo)

        if imgpath:
            plt.savefig(imgpath)

        if (plotgraph):
            plt.show()
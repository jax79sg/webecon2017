## generic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os
import glob
import cv2
import ujson as json
#import json
from collections import OrderedDict
from collections import Counter

import seaborn as sns
import math
import datetime
from collections import defaultdict

## Sklearn stuff
#from sklearn.model_selection import StratifiedKFold
from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF

## Keras stuffs
from keras import __version__ as keras_version
print('Keras version: {}'.format(keras_version))
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, ZeroPadding1D, AtrousConvolution1D, MaxPooling1D, AveragePooling1D,GlobalAveragePooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD,Adam
from keras import backend as K
from keras.models import Model, model_from_json
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping

## our functions
from ipinyouReader import * #name overlap
import ipinyouWriter
import BidModels
from Evaluator import * #name overlap
from BidModels import BidModelInterface
from BidPriceEstimator import BidEstimator

class CNNBidModel(BidModelInterface):

    def __init__(self, trainX,trainY,valX,valY,class_weights_mu=2.2,batch_size=32,total_epochs=20,learning_rate=0.0001):
        self.class_weights_mu = class_weights_mu
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.learning_rate = learning_rate
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.model_checkpoint_filepath = "./SavedCNNModels/Keras-CNN-chkpt-" + self.timestr + ".hdf5"
        self.model_config_filepath = "./SavedCNNModels/Keras-CNN-model-" + self.timestr + ".json"

        #class weights
        self.train_class_weight = self.__create_class_weight(trainY['click'])
        print("== Training class weights: {}".format(self.train_class_weight))

        ## Further process data into model input formats
        self.X_train = np.expand_dims(trainX.as_matrix(), axis=1)
        self.X_val = np.expand_dims(valX.as_matrix(), axis=1)

        ## Dimension params
        self.output_dim = nb_classes = 2
        self.input_dim = len(trainX.columns)
        self.Y_train=trainY['click'].as_matrix()
        self.Y_val = valY['click'].as_matrix()
        self.Y_click_train = to_categorical(self.Y_train, nb_classes)
        self.Y_click_val = to_categorical(self.Y_val, nb_classes)

    # def getBidPrice(self, testDF):
    #     # print("Setting up XGBoost for Test set")
    #     # xTest = testDF[self.X_column]
    #     #
    #     # xgdmat = xgb.DMatrix(xTest)
    #     # y_pred = self._model.predict(xgdmat)
    #     #
    #     # # y_pred = [1 if i >= 0.07 else 0 for i in y_pred]
    #     #
    #     # # bidprice = BidEstimator().linearBidPrice(y_pred, base_bid=220, avg_ctr=0.2)
    #     # bidprice = BidEstimator().linearBidPrice_mConfi(y_pred, base_bid=240, variable_bid=100, m_conf=0.8)
    #     #
    #     # bids = np.stack([testDF['bidid'], bidprice], axis=1)
    #     # bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])
    #
    #     return bids
    #
    #
    #

    def trainModel(self):

        print("=== Train click model")
        click_pred_model=self.__trainClickPredModel()
        # print("Setting up XGBoost for Training: X and Y")
        # xTrain = trainDF[self.X_column]
        # yTrain = trainDF[self.Y_column]
        #
        # # print(xTrain.columns)
        # print ("No of features in input matrix: %d" % len(xTrain.columns))
        #
        # optimised_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
        #              'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':3, 'learning_rate': 0.045}
        # xgdmat = xgb.DMatrix(xTrain, yTrain) # Create our DMatrix to make XGBoost more efficient
        # self._model = xgb.train(optimised_params, xgdmat, num_boost_round=432,  verbose_eval=False)
        #
        # print("Importance: ", self._model.get_fscore())
        # xgdmat = xgb.DMatrix(xTrain)
        # y_pred = self._model.predict(xgdmat)
        #
        # y_pred = [1 if i>0.12 else 0 for i in y_pred]
        #
        # ClickEvaluator().printClickPredictionScore(y_pred, yTrain)
        #
        # sns.set(font_scale = 1.5)
        # # xgb.plot_importance(self._model)
        #
        # fscore = self._model.get_fscore()
        # importance_frame = pd.DataFrame({'Importance': list(fscore.values()), 'Feature': list(fscore.keys())})
        # importance_frame.sort_values(by='Importance', inplace=True)
        # importance_frame.plot(kind='barh', x='Feature', figsize=(8, 8), color='orange')
        # plt.show()

    def __trainClickPredModel(self):

        ## define the model # https://keras.io/layers/convolutional/
        print("== define model")
        model = Sequential()
        model.add(Convolution1D(nb_filter=512,
                                filter_length=6,
                                border_mode='same', # 'valid', #The valid means there is no padding around input or feature map, while same means there are some padding around input or feature map, making the output feature map's size same as the input's
                                activation='relu',
                                input_shape=(1, self.input_dim),
                                init='lecun_uniform'
                                # lecun_uniform for both gets AUC: 0.865961 | (good split) AUC: 0.861570 with avg pool at end
                                # glorot_uniform for both gets AUC: 0.868817 | AUC: 0.863290  with avg pool at end
                                # he_uniform for both gets AUC: 0.868218 | AUC: 0.873585 with avg pool at end
                                ))

        # model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='same'))
        model.add(AveragePooling1D(pool_length=2, stride=None, border_mode='same'))
        # add a new conv1d on top
        model.add(Convolution1D(256, 3, border_mode='same', init='lecun_uniform', activation='relu', ))

        # model.add(AveragePooling1D(pool_length=2, stride=None, border_mode='same'))

        # # add a new conv1d on top AUC: 0.851369 with glorot uniform
        # model.add(Convolution1D(128, 3, border_mode='same',init='glorot_uniform',activation='relu',))

        # # apply an atrous convolution 1d with atrous rate 2 of length 3 to a sequence with 10 timesteps,
        # # with 64 output filters
        # model = Sequential()
        # model.add(AtrousConvolution1D(128, 3, atrous_rate=2, border_mode='same', input_shape=(1,input_dim)))

        # # add a new atrous conv1d on top
        # model.add(AtrousConvolution1D(64, 2, atrous_rate=2, border_mode='same'))

        # we use max pooling:
        # model.add(GlobalMaxPooling1D())
        model.add(GlobalAveragePooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(256, init='glorot_uniform'))
        model.add(Dropout(0.05))  # 0.1 seems good, but is it overfitting?
        model.add(Activation('relu'))

        # # We project onto a single unit output layer, and squash it with a sigmoid:
        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))
        # model.add(Dense(output_dim, input_dim=input_dim, activation='softmax',init='glorot_uniform'))
        model.add(Dense(self.output_dim, activation='softmax', init='glorot_uniform'))

        print(model.summary())
        #print(model.get_config())

        # write model to file
        with open(self.model_config_filepath,'w') as f:
            json.dump(model.to_json(),f)

        ### Compile model
        print("== Compile model")
        # optimizer = SGD(lr = self.learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
        optimizer = Adam(lr=self.learning_rate)

        # compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        ### Train the model
        # saves the model weights after each epoch if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath=self.model_checkpoint_filepath, verbose=1, save_best_only=True)

        # early stopping one class only
        earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

        ## For click pred only
        history = model.fit(self.X_train, [self.Y_click_train], \
                            batch_size=self.batch_size, nb_epoch=self.total_epochs,
                            validation_data=(self.X_val, [self.Y_click_val]),
                            class_weight=self.train_class_weight,
                            callbacks=[checkpointer, earlystopper],
                            verbose=2 # 0 silent, 1 verbose, 2 one log line per epoch
                            )  # TODO add callbacks, shuffle?
        self.click_pred_model=model

    def __create_class_weight(self,trainY):
        mu=self.class_weights_mu
        total = len(trainY)
        keys = trainY.unique()
        class_weight = dict()
        # print(total)

        for key in keys:
            # print(trainY['click'].value_counts()[key])
            score = math.log(mu * total / float(trainY.value_counts()[key]))
            # print(score)
            class_weight[key] = score if score > 1.0 else 1.0

        return class_weight

    def __load_trained_model(model,weights_path):
        #model = create_model()
        model.load_weights(weights_path)

    # def gridSearch(self, trainDF):
    #     # print("Setting up XGBoost for GridSearch: X and Y")
    #     # xTrain = trainDF[self.X_column]
    #     # yTrain = trainDF[self.Y_column]
    #     #
    #     # print(xTrain.columns)
    #     # print("No of features in input matrix: %d" % len(xTrain.columns))
    #     #
    #     # ## Setup Grid Search parameter
    #     # param_grid = {'max_depth': [3, 4],
    #     #               'min_child_weight': [2, 3],
    #     #               'subsample': [0.7, 0.8, 0.9],
    #     #               'learning_rate': [0.04, 0.045]
    #     #               }
    #     #
    #     # ind_params = {'n_estimators': 1000,
    #     #               'seed': 0,
    #     #               'colsample_bytree': 0.8,
    #     #               'objective': 'binary:logistic',
    #     #               'base_score': 0.5,
    #     #               'colsample_bylevel': 1,
    #     #               'gamma': 0,
    #     #               'max_delta_step': 0,
    #     #               'missing': None,
    #     #               'reg_alpha': 0,
    #     #               'reg_lambda': 1,
    #     #               'scale_pos_weight': 1,
    #     #               'silent': True,
    #     #               }
    #     #
    #     # optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
    #     #                              param_grid=param_grid,
    #     #                              scoring='accuracy',
    #     #                              cv=5,
    #     #                              n_jobs=-1,
    #     #                              error_score='raise')
    #     #
    #     # model = optimized_GBM.fit(xTrain, yTrain)
    #     # # check the accuracy on the training set
    #     # print("\n\nTraining acccuracy: %5.3f" % model.score(xTrain, yTrain))
    #     # y_pred = model.predict(xTrain)
    #     # p, r, f1, _ = metrics.precision_recall_fscore_support(yTrain, y_pred)
    #     # # print(p)
    #     # print("Number of 1: ", np.count_nonzero(y_pred))
    #     # print("Number of 0: ", len(y_pred) - np.count_nonzero(y_pred))
    #     # for i in range(len(p)):
    #     #     print("Precision: %5.3f \tRecall: %5.3f \tF1: %5.3f" % (p[i], r[i], f1[i]))
    #     # scores = optimized_GBM.grid_scores_
    #     # # print(type(scores))
    #     # for i in range(len(scores)):
    #     #     print(optimized_GBM.grid_scores_[i])
    #
    # def __estimateClick(self, df):
    #     # xTest = df[self.X_column]
    #     # print("No of features in input matrix: %d" % len(xTest.columns))
    #     #
    #     # xgdmat = xgb.DMatrix(xTest)
    #     # y_pred = self._model.predict(xgdmat)
    #
    #     return y_pred
    #
    #
    # # def tunelinearBaseBid(self, testDF):
    # #     print("Setting up XGBoost for Test set")
    # #     y_pred = self.__estimateClick(testDF)
    # #
    # #     # y_pred = [1 if i >= 0.07 else 0 for i in y_pred]
    # #
    # #     # avgCTR = np.count_nonzero(testDF.click) / testDF.shape[0]
    # #     myEvaluator = Evaluator.Evaluator()
    # #
    # #     bestCTR = -1
    # #     bestBidPrice = -1
    # #
    # #     print("y_pred mean: ", np.mean(y_pred))
    # #
    # #     x = np.arange(0.5, 0.9, 0.05)
    # #     # for i in x:
    # #     for i in range(220, 260):
    # #         print("================= i : ", i)
    # #         # bidprice = BidEstimator().linearBidPrice(y_pred, i, 0.2)
    # #         bidprice = BidEstimator().linearBidPrice_mConfi(y_pred, i, 90, 0.8)
    # #         bids = np.stack([testDF['bidid'], bidprice], axis=1)
    # #
    # #         bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])
    # #
    # #         # print("Estimated bid price: ", bids.bidprice.ix[0])
    # #
    # #         resultDict = myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validateDF)
    # #         myEvaluator.printResult()
    # #         ctr = resultDict['click'] / resultDict['won']
    # #
    # #         if ctr > bestCTR:
    # #             bestCTR = ctr
    # #             bestBidPrice = i
    # #
    # #     print("Best CTR: %.5f \nPrice: %d" %(bestCTR, bestBidPrice))
    # #
    # #
    # # def tuneConfidenceBaseBid(self, testDF):
    # #     print("Setting up XGBoost for Test set")
    # #     y_pred = self.__estimateClick(testDF)
    # #
    # #     y_pred = [1 if i >= 0.7 else 0 for i in y_pred]
    # #
    # #     # print("number of 1 here: ", sum(y_pred))
    # #     # avgCTR = np.count_nonzero(testDF.click) / testDF.shape[0]
    # #     myEvaluator = Evaluator.Evaluator()
    # #
    # #     bestCTR = -1
    # #     bestBidPrice = -1
    # #     for i in range(300, 301):
    # #         bidprice = BidEstimator().confidenceBidPrice(y_pred, -1, i)
    # #
    # #         # print("total bid price: ", sum(bidprice))
    # #         # print("total bid submitted: ", np.count_nonzero(bidprice))
    # #         # print("Number of $0 bid", bidprice.count(0))
    # #
    # #         bids = np.stack([testDF['bidid'], bidprice], axis=1)
    # #
    # #         bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])
    # #
    # #         # print("Estimated bid price: ", bids.bidprice.ix[0])
    # #
    # #         resultDict = myEvaluator.computePerformanceMetricsDF(25000 * 1000, bids, validateDF)
    # #         myEvaluator.printResult()
    # #         ctr = resultDict['click'] / resultDict['won']
    # #
    # #         if ctr > bestCTR:
    # #             bestCTR = ctr
    # #             bestBidPrice = i
    # #
    # #     print("Best CTR: %.5f \nPrice: %d" % (bestCTR, bestBidPrice))

    def predictClickProbs(self,X):
        click_probs = self.click_pred_model.predict(X)
        return click_probs

if __name__ == "__main__":
    ### Data
    TRAIN_FILE_PATH = "../dataset/train_cleaned_prune.csv" #"./data.pruned/train_cleaned_prune.csv"  # "../dataset/train.csv"
    VALIDATION_FILE_PATH = "../dataset/validation_cleaned_prune.csv" #"./data.pruned/validation_cleaned_prune.csv"  # "../dataset/validation.csv"
    TEST_FILE_PATH = "./data/test.csv"

    # Stratification
    NUM_K_FOLDS = 1
    SHUFFLE_INPUT = True
    RANDOM_SEED = None  # or int

    ### Weights
    CLASS_WEIGHTS_MU = 2.2  # 0.8 #0.15

    ### Features
    EXCLUDE_DOMAIN=False
    DOMAIN_KEEP_PROB=0.05

    ### Training
    BATCH_SIZE = 32
    TOTAL_EPOCHS = 20
    #DROPOUT_PROB = 0.2
    LEARNING_RATE = 0.0001  # adam #for SGD 0.003

    ### bidding strategy
    BASE_PRICE = 300

    ##########
    ## Load Dataset
    print("==== Reading in train and validation set...")
    trainReader = ipinyouReader.ipinyouReader(TRAIN_FILE_PATH)
    # trainData = trainReader.getTrainData()

    if VALIDATION_FILE_PATH:
        validationReader = ipinyouReader.ipinyouReader(VALIDATION_FILE_PATH)
        # validationData = validationReader.getTestData()

    ## onehot
    print("==== Convert to one-hot encoding...")
    trainOneHotData, trainY = trainReader.getOneHotData(exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
    print("Train set - No. of one-hot features: {}".format(len(trainOneHotData.columns)))

    if VALIDATION_FILE_PATH:
        valOneHotData, valY = validationReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
        print("Validation set - No. of one-hot features: {}".format(len(valOneHotData.columns)))

    print("==== Train CNN model...")
    bidmodel=CNNBidModel(trainOneHotData, trainY,valOneHotData,valY,class_weights_mu=CLASS_WEIGHTS_MU,batch_size=BATCH_SIZE,total_epochs=TOTAL_EPOCHS,learning_rate=LEARNING_RATE)
    bidmodel.trainModel()

    print("==== Predict clicks...")
    click_eval = ClickEvaluator()
    prob_click_train=bidmodel.predictClickProbs(bidmodel.X_train)
    if VALIDATION_FILE_PATH:
        prob_click_val = bidmodel.predictClickProbs(bidmodel.X_val)

        print("== Click prob distributions...")
        # click=1 prediction as click = =1 probabilities
        click1 = prob_click_val[bidmodel.Y_click_val[:, 1].astype(bool), 1]
        n, bins, patches = click_eval.clickProbHistogram(pred_prob=click1,color='g',title='Predicted probabilities for clicks=1',imgpath="./SavedCNNModels/Keras-CNN-click1-" + bidmodel.timestr + ".jpg",showGraph=False)

        # click=0 prediction as click=1 probabilities
        click0 = prob_click_val[bidmodel.Y_click_val[:, 0].astype(bool), 1]
        n, bins, patches = click_eval.clickProbHistogram(pred_prob=click0,color='r',title='Predicted probabilities for clicks=0',imgpath="./SavedCNNModels/Keras-CNN-click0-" + bidmodel.timestr + ".jpg",showGraph=False)

        print("== ROC for click model...")
        roc_auc = click_eval.clickROC(bidmodel.Y_click_val[:, 1], prob_click_val[:, 1],imgpath="./SavedCNNModels/Keras-CNN-ROC-" + bidmodel.timestr + ".jpg",showGraph=False)

        print("== Find pred_threshold best f1 score...")
        # Gold
        Y_1_click_train = bidmodel.Y_click_train[:, 1]
        Y_1_click_val = bidmodel.Y_click_val[:, 1]

        for pred_threshold in np.arange(0.0,1.05,0.05):
            print("= pred_threshold = {}".format(pred_threshold))
            # Pick pred_threshold for click=1 as click=1
            pred_1_click_train = np.greater_equal(prob_click_train[:, 1], pred_threshold).astype(int)
            pred_1_click_val = np.greater_equal(prob_click_val[:, 1], pred_threshold).astype(int)

            # Validation
            click_precision, click_recall, click_f1score = \
                click_eval.printClickPredictionScore(y_Pred=pred_1_click_val, y_Gold=Y_1_click_val)



#TODO: slotformat, adexhcange potentially for imputing
# a['slotformat'].value_counts()
# 0 5325
# 1 2363
# Na 2091
# 5 150


# a['adexchange'].value_counts()
# 3 3273
# 2 3020
# 1 2925
# null 402

# domain nulls ??
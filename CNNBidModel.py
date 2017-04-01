## generic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os
import glob
# import cv2
import ujson as json
#import json
from collections import OrderedDict
from collections import Counter
import re

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

    def __init__(self, trainX,trainY,valX=None,valY=None,testX=None,testbidids=None,class_weights_mu=2.2,batch_size=32,total_epochs=20,learning_rate=0.0001,shuffle=False,reserve_val=True):
        self.class_weights_mu = class_weights_mu
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.learning_rate = learning_rate
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.model_checkpoint_filepath = "./SavedCNNModels/Keras-CNN-chkpt-" + self.timestr + ".hdf5"
        self.model_config_filepath = "./SavedCNNModels/Keras-CNN-model-" + self.timestr + ".json"
        self.bids_output_filepath = "./SavedCNNModels/Keras-CNN-testbids-" + self.timestr + ".csv"
        self.bids_tuning_perf_filepath = "./SavedCNNModels/Keras-CNN-bidtuning-" + self.timestr + ".csv"
        self.shuffle=shuffle
        self.reserve_val = reserve_val

        #class weights
        self.train_class_weight = self.__create_class_weight(trainY['click'])
        print("== Training class weights: {}".format(self.train_class_weight))

        ## Further process data into model input formats
        self.X_train = np.expand_dims(trainX.as_matrix(), axis=1)
        self.X_val = np.expand_dims(valX.as_matrix(), axis=1)
        self.X_test = np.expand_dims(testX.as_matrix(), axis=1)
        ## Dimension params
        self.output_dim = nb_classes = 2
        self.input_dim = len(trainX.columns)

        ## Y process
        self.Y_train=trainY['click'].as_matrix()
        self.Y_click_train = to_categorical(self.Y_train, nb_classes)
        self.Y_val = valY['click'].as_matrix()
        self.Y_click_val = to_categorical(self.Y_val, nb_classes)

        ## Panda frames for bidids
        self.bidids_test = testbidids
        self.gold_val = valY

    def getBidPrice(self,y_prob,bidids,base_bid,slotprices,pred_thresh=0.5):
        avg_ctr = ClickEvaluator().compute_avgCTR(self.Y_train)
        print("Train avgCTR = {}".format(avg_ctr))

        bid_estimator = BidEstimator()
        #bids = bid_estimator.linearBidPrice(y_pred, 50, avg_ctr)
        #TODO: could add option for alternate  bid strats
        bids = bid_estimator.linearBidPrice_variation(y_prob,base_bid,avg_ctr,slotprices,pred_thresh)
        print(bids)
        # format bids into bidids pandas frame
        bids_df = bidids.copy()
        bids_df['bidprice'] = bids
        ipinyouWriter.ResultWriter().writeResult(self.bids_output_filepath, bids_df)
        return bids_df

    def gridSearchBidPrice(self, y_prob, slotprices):
        print("=== Get best bid prices")
        avg_ctr = ClickEvaluator().compute_avgCTR(self.Y_train)
        print("Train avgCTR = {}".format(avg_ctr))

        bid_estimator = BidEstimator()
        # TODO: could add option for alternate  bid strats
        best_pred_thresh, best_base_bid, perf_df = bid_estimator.gridSearch_bidPrice(y_prob, avg_ctr, slotprices,self.gold_val,bidpriceest_model='linearBidPrice')
        ipinyouWriter.ResultWriter().writeResult(re.sub('.csv','-linearBidPrice.csv',self.bids_tuning_perf_filepath), perf_df) #
        best_pred_thresh, best_base_bid, perf_df = bid_estimator.gridSearch_bidPrice(y_prob, avg_ctr, slotprices,self.gold_val,bidpriceest_model='linearBidPrice_variation')
        ipinyouWriter.ResultWriter().writeResult(re.sub('.csv','-linearBidPrice_variation.csv',self.bids_tuning_perf_filepath), perf_df)


        return best_pred_thresh,best_base_bid

    def trainModel(self):

        print("=== Train click model")
        click_pred_model=self.__trainClickPredModel()

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
        # model.add(Convolution1D(256, 3, border_mode='same', init='glorot_uniform', activation='relu', )) #on the fence about effect

        #model.add(AveragePooling1D(pool_length=2, stride=None, border_mode='same')) #worse if added

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
        model.add(Dropout(0.2))  # 0.1 seems good, but is it overfitting?
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

        self.click_pred_model = model

        #actually run training
        self.trainClickPredModelRunTraining()

    def trainClickPredModelRunTraining(self):
        ### Train the model
        # saves the model weights after each epoch if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath=self.model_checkpoint_filepath, verbose=1, save_best_only=True)

        # early stopping one class only
        earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

        ## For click pred only
        if self.reserve_val: #we are reserving validation set for later
            # do val split on training data automatically
            print("== Training model using training data for validation split (validation set will be held out)")
            history = self.click_pred_model.fit(self.X_train, [self.Y_click_train], \
                                batch_size=self.batch_size, nb_epoch=self.total_epochs,
                                validation_split=0.1,
                                shuffle=self.shuffle,
                                class_weight=self.train_class_weight,
                                callbacks=[checkpointer, earlystopper],
                                verbose=2  # 0 silent, 1 verbose, 2 one log line per epoch
                                )  # TODO add callbacks, shuffle?
        else:
            print("== Training model using training + validation data for validation")
            history = self.click_pred_model.fit(self.X_train, [self.Y_click_train], \
                                batch_size=self.batch_size, nb_epoch=self.total_epochs,
                                validation_data=(self.X_val, [self.Y_click_val]),
                                shuffle=self.shuffle,
                                class_weight=self.train_class_weight,
                                callbacks=[checkpointer, earlystopper],
                                verbose=2  # 0 silent, 1 verbose, 2 one log line per epoch
                                )  # TODO add callbacks, shuffle?



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

    def loadSavedModel(self,weights_path):
        #model = create_model()
        self.click_pred_model.load_weights(weights_path)

    def saveModel(self):
        print("saveModel not implemented as built into Keras")

    def predictClickProbs(self,X):
        click_probs = self.click_pred_model.predict(X)
        return click_probs

if __name__ == "__main__":
    ### Data
    #TRAIN3_FILE_PATH = "./data.pruned/train_cleaned_prune.csv"  #"./data/medium_train_cleaned_prune.csv" #"../dataset/train_cleaned_prune.csv" #"./data/larger_train_cleaned_prune.csv" #"../dataset/train_cleaned_prune.csv" #"./data.pruned/train_cleaned_prune.csv"  # "../dataset/train.csv"
    #TRAIN_FILE_PATH = "../dataset/train_cleaned_prune.csv" #"./data/larger_train_cleaned_prune.csv" #"../dataset/train_cleaned_prune.csv" #"./data.pruned/train_cleaned_prune.csv"  # "../dataset/train.csv"
    #TRAIN_FILE_PATH = "./data/medium_train_cleaned_prune.csv"  # "./data/larger_train_cleaned_prune.csv" #"../dataset/train_cleaned_prune.csv" #"./data.pruned/train_cleaned_prune.csv"  # "../dataset/train.csv"
    #TRAIN2_FILE_PATH = "" #./data/small_train_cleaned_prune.csv"  # "./data/larger_train_cleaned_prune.csv" #"../dataset/train_cleaned_prune.csv" #"./data.pruned/train_cleaned_prune.csv"  # "../dataset/train.csv"
    TRAIN_FILE_PATH = "./data.final/train1_cleaned_prune.csv"
    TRAIN2_FILE_PATH = "./data.final/train2_cleaned_prune.csv"
    TRAIN3_FILE_PATH = "./data.final/train3_cleaned_prune.csv"
    VALIDATION_FILE_PATH = "./data.final/validation_cleaned.csv" #"" #"../dataset/validation_cleaned_prune.csv" #"./data.pruned/validation_cleaned_prune.csv"  # "../dataset/validation.csv" "" #
    TEST_FILE_PATH = "./data.final/test.csv"

    ### preproc trained Data
    PREPROC_X_TRAIN_FILE_PATH = "" #"./data/onehot_X_nodomain_merged_train_validation_cleaned.csv"  #"./data/onehot_X_merged_train_validation_cleaned.csv" #"./data.pruned/train_cleaned_prune.csv"  # "../dataset/train.csv"
    PREPROC_Y_TRAIN_FILE_PATH = "" #"./data/onehot_Y_nodomain_merged_train_validation_cleaned.csv" #"./data/onehot_Y_merged_train_validation_cleaned.csv"
    PREPROC_TEST_FILE_PATH = "" #"./data/test.csv"

    # Stratification
    NUM_K_FOLDS = 1
    SHUFFLE_INPUT = True
    RANDOM_SEED = None  # or int
    RESERVE_VAL = True # Reserve validation set for further tuning, in CTR training just use train splits
    #VALIDATION_SPLIT=0.2

    ### Weights
    CLASS_WEIGHTS_MU = 2.2  # 0.8 #0.15

    ### Features
    EXCLUDE_DOMAIN=False
    DOMAIN_KEEP_PROB=0.05 #1.0

    ### Training
    BATCH_SIZE = 32
    TOTAL_EPOCHS = 20
    #DROPOUT_PROB = 0.2
    LEARNING_RATE = 0.0001  # adam #for SGD 0.003

    ### bidding strategy
    #BASE_PRICE = 300

    ##########
    ## Load Dataset
    if PREPROC_X_TRAIN_FILE_PATH and PREPROC_Y_TRAIN_FILE_PATH:
        print("==== Reading in pre-processed trainset...")
        XtrainReader = ipinyouReader.ipinyouReader(PREPROC_X_TRAIN_FILE_PATH)
        trainOneHotData = XtrainReader.getDataFrame()
        YtrainReader = ipinyouReader.ipinyouReader(PREPROC_Y_TRAIN_FILE_PATH)
        trainY = YtrainReader.getDataFrame()
        print("Train set - No. of one-hot features: {}".format(len(trainOneHotData.columns)))
    else:
        print("==== Reading in train set...")
        print("Train file: {}".format(TRAIN_FILE_PATH))
        trainReader = ipinyouReader.ipinyouReader(TRAIN_FILE_PATH)
        # trainData = trainReader.getTrainData()

        ## onehot
        print("== Convert to one-hot encoding...")
        trainOneHotData, trainY = trainReader.getOneHotData(exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
        print("Train set - No. of one-hot features: {}".format(len(trainOneHotData.columns)))

        if VALIDATION_FILE_PATH:
            print("==== Reading in validation set...")
            print("Validation file: {}".format(VALIDATION_FILE_PATH))
            validationReader = ipinyouReader.ipinyouReader(VALIDATION_FILE_PATH)
            valOneHotData, valY = validationReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
            print("Validation set - No. of one-hot features: {}".format(len(valOneHotData.columns)))

        if TEST_FILE_PATH:
            print("==== Reading in test set...")
            print("Test file: {}".format(TEST_FILE_PATH))
            testReader = ipinyouReader.ipinyouReader(TEST_FILE_PATH)
            testOneHotData,testbidids = testReader.getOneHotData(train_cols=trainOneHotData.columns.get_values().tolist(),exclude_domain=EXCLUDE_DOMAIN,domain_keep_prob=DOMAIN_KEEP_PROB)#0.05)
            print("Test set - No. of one-hot features: {}".format(len(testOneHotData.columns)))


    if TRAIN2_FILE_PATH:
        print("Train2 file: {}".format(TRAIN2_FILE_PATH))
        train2Reader = ipinyouReader.ipinyouReader(TRAIN2_FILE_PATH)
        train2OneHotData, train2Y = train2Reader.getOneHotData(train_cols=[],#trainOneHotData.columns.get_values().tolist(),
                                                             exclude_domain=EXCLUDE_DOMAIN,
                                                             domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
        print("Train2 set - No. of one-hot features: {}".format(len(train2OneHotData.columns)))

        if VALIDATION_FILE_PATH:
            print("==== Reading in validation set...")
            print("Validation file: {}".format(VALIDATION_FILE_PATH))
            validation2Reader = ipinyouReader.ipinyouReader(VALIDATION_FILE_PATH)
            val2OneHotData, val2Y = validation2Reader.getOneHotData(
                train_cols=train2OneHotData.columns.get_values().tolist(), exclude_domain=EXCLUDE_DOMAIN,
                domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
            print("Validation set - No. of one-hot features: {}".format(len(val2OneHotData.columns)))

        if TEST_FILE_PATH:
            print("==== Reading in test set...")
            print("Test file: {}".format(TEST_FILE_PATH))
            test2Reader = ipinyouReader.ipinyouReader(TEST_FILE_PATH)
            test2OneHotData, test2bidids = test2Reader.getOneHotData(
                train_cols=train2OneHotData.columns.get_values().tolist(), exclude_domain=EXCLUDE_DOMAIN,
                domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
            print("Test set - No. of one-hot features: {}".format(len(test2OneHotData.columns)))

    if TRAIN3_FILE_PATH:
        print("Train3 file: {}".format(TRAIN3_FILE_PATH))
        train3Reader = ipinyouReader.ipinyouReader(TRAIN3_FILE_PATH)
        train3OneHotData, train3Y = train3Reader.getOneHotData(train_cols=[],#trainOneHotData.columns.get_values().tolist(),
                                                             exclude_domain=EXCLUDE_DOMAIN,
                                                             domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
        print("Train3 set - No. of one-hot features: {}".format(len(train3OneHotData.columns)))

        if VALIDATION_FILE_PATH:
            print("==== Reading in validation set...")
            print("Validation file: {}".format(VALIDATION_FILE_PATH))
            validation3Reader = ipinyouReader.ipinyouReader(VALIDATION_FILE_PATH)
            val3OneHotData, val3Y = validation3Reader.getOneHotData(
                train_cols=train3OneHotData.columns.get_values().tolist(), exclude_domain=EXCLUDE_DOMAIN,
                domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
            print("Validation set - No. of one-hot features: {}".format(len(val3OneHotData.columns)))

        if TEST_FILE_PATH:
            print("==== Reading in test set...")
            print("Test file: {}".format(TEST_FILE_PATH))
            test3Reader = ipinyouReader.ipinyouReader(TEST_FILE_PATH)
            test3OneHotData, test3bidids = test3Reader.getOneHotData(
                train_cols=train3OneHotData.columns.get_values().tolist(), exclude_domain=EXCLUDE_DOMAIN,
                domain_keep_prob=DOMAIN_KEEP_PROB)  # 0.05)
            print("Test set - No. of one-hot features: {}".format(len(test3OneHotData.columns)))


    print("==== Train CNN model1...")
    bidmodel = CNNBidModel(trainOneHotData, trainY,valOneHotData,valY, testOneHotData,testbidids,class_weights_mu=CLASS_WEIGHTS_MU,batch_size=BATCH_SIZE, total_epochs=TOTAL_EPOCHS, learning_rate=LEARNING_RATE, shuffle=SHUFFLE_INPUT,reserve_val=RESERVE_VAL)
    bidmodel.trainModel()
    print("== Reload to best weights saved...")
    bidmodel.loadSavedModel(bidmodel.model_checkpoint_filepath)

    if TRAIN2_FILE_PATH:
        print("==== Train CNN model2...")
        bidmodel2 = CNNBidModel(train2OneHotData, train2Y, val2OneHotData, val2Y, test2OneHotData, test2bidids,
                               class_weights_mu=CLASS_WEIGHTS_MU, batch_size=BATCH_SIZE, total_epochs=TOTAL_EPOCHS,
                               learning_rate=LEARNING_RATE, shuffle=SHUFFLE_INPUT, reserve_val=RESERVE_VAL)
        bidmodel2.trainModel()
        print("== Reload to best weights saved...")
        bidmodel2.loadSavedModel(bidmodel2.model_checkpoint_filepath)

    if TRAIN3_FILE_PATH:
        print("==== Train CNN model3...")
        bidmodel3 = CNNBidModel(train3OneHotData, train3Y, val3OneHotData, val3Y, test3OneHotData, test3bidids,
                                class_weights_mu=CLASS_WEIGHTS_MU, batch_size=BATCH_SIZE, total_epochs=TOTAL_EPOCHS,
                                learning_rate=LEARNING_RATE, shuffle=SHUFFLE_INPUT, reserve_val=RESERVE_VAL)
        bidmodel3.trainModel()
        print("== Reload to best weights saved...")
        bidmodel3.loadSavedModel(bidmodel3.model_checkpoint_filepath)

    #if TRAIN2_FILE_PATH:
    # TODO: for this to work properly need the usertag features from here too truly?Whether it's as a seperate model
    # print("==== Train2 CNN model...")
    # bidmodel2 = CNNBidModel(train2OneHotData, train2Y,valOneHotData,valY, testOneHotData,testbidids,class_weights_mu=CLASS_WEIGHTS_MU,batch_size=BATCH_SIZE, total_epochs=TOTAL_EPOCHS, learning_rate=LEARNING_RATE, shuffle=SHUFFLE_INPUT,reserve_val=RESERVE_VAL)
    # bidmodel.X_train = bidmodel2.X_train
    # bidmodel.Y_click_train = bidmodel2.Y_click_train
    # bidmodel.X_val = bidmodel2.X_val
    # bidmodel.Y_click_val = bidmodel2.Y_click_val
    # bidmodel.trainClickPredModelRunTraining()
    # print("== Reload to best weights saved...")
    # bidmodel.loadSavedModel(bidmodel.model_checkpoint_filepath)

    print("==== Predict clicks...")
    click_eval = ClickEvaluator()
    if RESERVE_VAL:
        prob_click_train1 = bidmodel.predictClickProbs(bidmodel.X_train)
        prob_click_val1 = bidmodel.predictClickProbs(bidmodel.X_val)
        prob_click_test1 = bidmodel.predictClickProbs(bidmodel.X_test)

        if TRAIN2_FILE_PATH:
            prob_click_train2 = bidmodel2.predictClickProbs(bidmodel2.X_train)
            prob_click_val2 = bidmodel2.predictClickProbs(bidmodel2.X_val)
            prob_click_test2 = bidmodel2.predictClickProbs(bidmodel2.X_test)
        if TRAIN3_FILE_PATH:
            prob_click_train3 = bidmodel3.predictClickProbs(bidmodel3.X_train)
            prob_click_val3 = bidmodel3.predictClickProbs(bidmodel3.X_val)
            prob_click_test3 = bidmodel3.predictClickProbs(bidmodel3.X_test)

        if not TRAIN2_FILE_PATH and not TRAIN3_FILE_PATH:
            prob_click_train = prob_click_train1
            prob_click_val=prob_click_val1
            prob_click_test = prob_click_test1
        elif not TRAIN3_FILE_PATH:
            prob_click_train = np.add(prob_click_train1,prob_click_train2)/2
            prob_click_val = np.add(prob_click_val1,prob_click_val2)/2
            prob_click_test = np.add(prob_click_test1, prob_click_test2) / 2
        else:
            prob_click_train = np.add(np.add(prob_click_train1, prob_click_train2),prob_click_train3) / 3
            prob_click_val = np.add(np.add(prob_click_val1, prob_click_val2),prob_click_val3) / 3
            prob_click_test = np.add(np.add(prob_click_test1, prob_click_test2),prob_click_test3) / 3

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
        slotprices_val = validationReader.getDataFrame()['slotprice'].as_matrix().astype(int)
        pred_thresh,base_bid=bidmodel.gridSearchBidPrice(prob_click_val[:,1],slotprices_val)

        print("== Get bid price for test data using pred_thresh {0:.2f} and base_bid {1} ...".format(pred_thresh,base_bid))
        slotprices_test = testReader.getDataFrame()['slotprice'].as_matrix().astype(int)
        bids_df=bidmodel.getBidPrice(prob_click_test[:,1], bidmodel.bidids_test, base_bid, slotprices_test, pred_thresh)


        # for pred_threshold in np.arange(0.1,1.00,0.1): #np.arange(0.05,1.00,0.05):
        #     print("= pred_threshold = {}".format(pred_threshold))
        #     # Pick pred_threshold for click=1 as click=1
        #     pred_1_click_train = np.greater_equal(prob_click_train[:, 1], pred_threshold).astype(int)
        #     pred_1_click_val = np.greater_equal(prob_click_val[:, 1], pred_threshold).astype(int)
        #
        #     # on Train
        #     print("True Train CTR: {0:.4f}".format(click_eval.compute_avgCTR(Y_1_click_train)))
        #     print("Pred Train CTR: {0:.4f}".format(click_eval.compute_avgCTR(pred_1_click_train)))
        #
        #     # on Validation
        #     print("True Val   CTR: {0:.4f}".format(click_eval.compute_avgCTR(Y_1_click_val)))
        #     print("Pred Val   CTR: {0:.4f}".format(click_eval.compute_avgCTR(pred_1_click_val)))
        #
        #     # Validation
        #     click_precision, click_recall, click_f1score = \
        #         click_eval.printClickPredictionScore(y_Pred=pred_1_click_val, y_Gold=Y_1_click_val)
        #     base_bid=80
        #     bids=bidmodel.getBidPrice(prob_click_val[:,1],base_bid,slotprices_val,pred_threshold)




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
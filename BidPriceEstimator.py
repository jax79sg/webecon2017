import numpy as np
import pandas as pd
from Evaluator import * #name overlap

class BidEstimator():

    def linearBidPrice(self, y_pred, base_bid, avg_ctr):
        '''
        Based on the formula
            bid = base_bid x (pCTR/avgCTR)

        p_ctr = prob of 1 from model
        avg_ctr = np.count_nonzero(testset)/len(testset)


        :param y_pred: Array of prediction based on 1 and 0. 1 means click
        :param base_bid: A integer value between 1 to 1000 or anything
        :param avg_ctr:
        :return:
        '''
        # print("y_pred: ", y_pred[0:10])
        #bids = [base_bid * (i/avg_ctr) for i in y_pred]
        bids = base_bid * (y_pred / avg_ctr)
        # print("ave bid: ", np.mean(bids))
        # print(bids[0:10])

        return bids

    def linearBidPrice_mConfi(self, y_pred, base_bid, variable_bid, m_conf):
        '''
        Based on the formula
            bid = base_bid + variable_bid ( (y_pred[i] - m_conf) / (1-m_conf) ) if y_pred[i] > m_conf
            -1                                                                  else

        1. Only bid when confidence is higher than m_conf
        2. Price to bid is depend on the confidence level

        :param y_pred: Array of prediction based on 1 and 0. 1 means click
        :param base_bid: A integer value between 1 to 1000 or anything
        :param variable_bid: Range of variable bid
        :param m_conf: Minimum confidence level
        :return:
        '''
        # bids = [base_bid * (i/avg_ctr) if i>min_confidence else -1 for i in y_pred]
        bids = [base_bid + (((i-m_conf) / (1-m_conf))*variable_bid) if i > m_conf else -1 for i in y_pred]
        # print("y_pred: ", y_pred[0:20])
        # print(bids[0:20])

        return bids

    def confidenceBidPrice(self, y_pred, base_bid, max_variable_bid):
        '''
        Bid higher when the click prediction is more confident

        Bid = base_bid + (Confidence for the impression * max_variable_bid)

        :param y_pred: Array of prediction in their original confidence
        :param base_bid: The base price to bid
        :param max_variable_bid: Variable component
        :return: bids
        '''
        bid = []
        for i in y_pred:
            bid.append(base_bid+int(i*max_variable_bid))

        return bid

    def linearBidPrice_variation(self, y_prob, base_bid, avg_ctr, slotprices, prune_thresh):
        # 1. Get linear bid
        # 2. Increase bid to min slot price
        # 3. Prune those below pred thresh

        # 1. Get linear bid
        initial_bids = self.linearBidPrice(y_prob, base_bid, avg_ctr)
        print('inital bid total               : {}'.format(int(initial_bids.sum())))

        # 2. Increase bid to min slot price + bit of fudge
        bids_reserve_price_met = np.maximum(initial_bids, slotprices * 1.1)
        print('bid total post-reserve match   : {}'.format(int(bids_reserve_price_met.sum())))
        ## alternative where we prune instead:
        # bids_reserve_price_met = np.ma.masked_less(initial_bids, slotprices)  # mask out values less than
        # np.ma.set_fill_value(bids_reserve_price_met, 0.0)
        # # print(bids_reserve_price_met.filled()) # return with filled value i.e masked out values become 0
        # print('bid total post-reserve prune   : {}'.format(int(bids_reserve_price_met.filled().sum())))

        # 3. Prune those below pred thresh
        prune_tresh_mask = np.less(y_prob, prune_thresh)
        # print(prune_tresh_mask)
        # bids_click_thresh_mask = np.ma.masked_array(bids_reserve_price_met.filled(), prune_tresh_mask) #this is for the alternative step 2.
        bids_click_thresh_mask = np.ma.masked_array(bids_reserve_price_met, prune_tresh_mask)
        np.ma.set_fill_value(bids_click_thresh_mask, -1.0) #-1.0 as some payprice == 0 :|
        # print(bids_click_thresh_mask.filled()) # return with filled value i.e masked out values become 0
        print('bid total post-prob {0:.2f} prune  : {1}'.format(prune_thresh, int(bids_click_thresh_mask.filled().sum())))

        return bids_click_thresh_mask.filled().astype(int)

    def thresholdSigmoid(self, predOneProb,noBidThreshold=0.4,minBid=220,bidRange=200,sigmoidDegree=-30):
        """
        To capture the essence of limited budget, capitalising on higher chances of clicks,
        bidding from a base and scaling the bid along the probabilities of clicks in a non-linear manner.
        :param predOneProb: Probability of click==1
        :param noBidThreshold: if predOneProb falls below this threshold, there would be no bid. (E.g. bid=-1)
        :param minBid: A min bid price for every bid
        :param bidRange: A bid range
        :param sigmoidDegree: For the sigmoid function. -10 less compressed (Relaxed), -100 very compressed (Threshold)
        :return: Array of bids
        """
        def sigmoid(x, threshold=None, sigmoiddegree=-30):
            """
            Internal function. Should not be adjusted.
            :param x:
            :param threshold:
            :param sigmoiddegree:
            :return:
            """
            sigmoidthreshold = -0.2 - threshold #Fixed at -0.2 base threshold for simplicity.
            a = 1 / (1 + np.exp(sigmoiddegree * (x + sigmoidthreshold)))
            return a

        def getBidPrice(clickProb, noBidThreshold, bidRange, minBid, sigmoiddegree):
            """
            Internal function, should not be adjusted.
            :param clickProb: Single click prob (Not an array)
            :return:
            """
            bid = 0
            if (clickProb > noBidThreshold):
                bid = (sigmoid(clickProb, noBidThreshold, sigmoiddegree)) * bidRange + minBid
            return bid

        f = np.vectorize(getBidPrice)
        bids = f(predOneProb, noBidThreshold=noBidThreshold, bidRange=bidRange, minBid=minBid, sigmoiddegree=sigmoidDegree)
        return bids

    def gridSearch_bidPrice(self,y_prob, avg_ctr, slotprices,gold_df,budget=6250000,bidpriceest_model='linearBidPrice',):
        # TODO this could be generalised to other models too.
        performance_list = []
        if bidpriceest_model == 'linearBidPrice':
            for base_bid in range(50, 300, 10):  # range(100,300,10):
                print(" = base_bid = {}".format(base_bid))
                bids = self.linearBidPrice(y_prob, base_bid, avg_ctr)
                # format bids into bidids pandas frame
                est_bids_df = gold_df[['bidid']].copy()
                est_bids_df['bidprice'] = bids
                myEvaluator = Evaluator()
                myEvaluator.computePerformanceMetricsDF(budget, est_bids_df, gold_df, verbose=False)
                # myEvaluator.printResult()
                myEvaluator.resultDict['base_bid'] = base_bid
                # print(myEvaluator.resultDict)
                performance_list += [myEvaluator.resultDict]
        elif bidpriceest_model == 'linearBidPrice_variation':
            for pred_threshold in np.arange(0.05,1.00,0.05): #np.arange(0.1,1.00,0.1): #
                print("== pred_threshold = {}".format(pred_threshold))
                for base_bid in range(50,500,10):#range(100,300,10):
                    print(" = base_bid = {}".format(base_bid))
                    bids=self.linearBidPrice_variation(y_prob,base_bid,avg_ctr, slotprices,pred_threshold)
                    # format bids into bidids pandas frame
                    est_bids_df = gold_df[['bidid']].copy()
                    est_bids_df['bidprice'] = bids
                    myEvaluator = Evaluator()
                    myEvaluator.computePerformanceMetricsDF(budget, est_bids_df, gold_df, verbose=True)
                    #myEvaluator.printResult()
                    myEvaluator.resultDict['pred_threshold']=pred_threshold
                    myEvaluator.resultDict['base_bid']=base_bid
                    #print(myEvaluator.resultDict)
                    performance_list+=[myEvaluator.resultDict]
        elif bidpriceest_model == 'linearBidPrice_mConfi':
            myEvaluator = Evaluator()

            total_gold_clicks = len(gold_df[gold_df['click'] == 1])

            basebid_grid = np.arange(20, 221, 20)
            variable_grid = np.arange(20, 121, 20)
            confi_grid = np.arange(0.1, 0.91, 0.1)
            # basebid_grid = np.arange(220, 230, 5)
            # variable_grid = np.arange(0, 20, 10)
            # confi_grid = np.arange(0.85, 0.95, 0.05)

            performance_list = []
            for basebid in basebid_grid:
                for variable in variable_grid:
                    for confi in confi_grid:
                        # bidprice = BidEstimator().linearBidPrice(y_pred, i, 0.2)
                        bidprice = self.linearBidPrice_mConfi(y_prob, basebid, variable, confi)
                        bids = np.stack([gold_df['bidid'], bidprice], axis=1)
                        bids = pd.DataFrame(bids, columns=['bidid', 'bidprice'])

                        resultDict = myEvaluator.computePerformanceMetricsDF(budget, bids, gold_df)
                        resultDict['base_bid'] = basebid
                        resultDict['pred_threshold'] = confi
                        resultDict['variable_bid'] = variable

                        # Store result Dict
                        performance_list.append(resultDict)

        elif bidpriceest_model == 'thresholdsigmoid':
            #Variables to tune
            #noBidThreshold=0.4,minBid=220,bidRange=200,sigmoidDegree=-30 (Baselines)
            counter=0
            # noBidThresholds=np.linspace(0.4,0.95,10)
            # minBids=range(200, 220, 10)
            # bidRanges=range(50, 250, 50)
            # sigmoidDegrees=[-20,-30,-40]

            noBidThresholds=np.linspace(0.2,0.95,10)
            minBids=range(100, 300, 50)
            bidRanges=range(50, 300, 50)
            sigmoidDegrees=[-10,-20,-30,-40,-50]


            for noBidThreshold in noBidThresholds:
                for minBid in minBids:
                    for bidRange in bidRanges:
                        for sigmoidDegree in sigmoidDegrees:
                            print("Pass ",counter," of ", len(noBidThresholds)*len(minBids)*len(bidRanges)*len(sigmoidDegrees))

                            bids = self.thresholdSigmoid(predOneProb=y_prob,noBidThreshold=noBidThreshold,minBid=minBid,bidRange=bidRange,sigmoidDegree=sigmoidDegree)
                            est_bids_df = gold_df[['bidid']].copy()
                            est_bids_df['bidprice'] = bids
                            myEvaluator = Evaluator()
                            myEvaluator.computePerformanceMetricsDF(budget, est_bids_df, gold_df, verbose=True)
                            myEvaluator.resultDict['noBidThreshold'] = noBidThreshold
                            myEvaluator.resultDict['minBid'] = minBid
                            myEvaluator.resultDict['bidRange'] = bidRange
                            myEvaluator.resultDict['sigmoidDegree'] = sigmoidDegree
                            performance_list += [myEvaluator.resultDict]
                            counter=counter+1

        else:
            print("bidpriceest_model '{}' not implemented yet".format(bidpriceest_model))

        ## stick performance metrics into pandas frame
        if bidpriceest_model == 'linearBidPrice':
            performance_pd = pd.DataFrame(performance_list,columns=['base_bid', 'won', 'click', 'spend', 'trimmed_bids', 'CTR', 'CPM', 'CPC'])
        elif bidpriceest_model == 'linearBidPrice_variation':
            performance_pd = pd.DataFrame(performance_list, columns=['base_bid', 'pred_threshold', 'won', 'click', 'spend','trimmed_bids', 'CTR', 'CPM', 'CPC'])
        elif bidpriceest_model == 'linearBidPrice_mConfi':
            performance_pd = pd.DataFrame(performance_list,
                                          columns=['base_bid', 'pred_threshold', 'variable_bid', 'won', 'click', 'spend',
                                                   'trimmed_bids', 'CTR', 'CPM', 'CPC'])
        elif bidpriceest_model == 'thresholdsigmoid':
            performance_pd = pd.DataFrame(performance_list, columns=['noBidThreshold','minBid','bidRange','sigmoidDegree', 'won', 'click', 'spend','trimmed_bids', 'CTR', 'CPM', 'CPC'])

        print("GRID SEARCH PERF TABLE")
        print(performance_pd)

        ## Determine 'best' params from 'CTR', 'click'/total_gold_clicks, (Removed: 1-'spend'/max_budget)
        total_gold_clicks = len(gold_df[gold_df['click'] == 1])
        perf_score = performance_pd.apply(lambda x: x['CTR'] * 0.6 + (x['click']/ total_gold_clicks)*0.4 , axis=1) #+ (1-x['spend']/budget) * 0.2
        performance_pd['blended_score'] = perf_score
        print("GRID SEARCH PERF SCORE")
        print(perf_score)
        best_idx=perf_score.idxmax(axis=0)

        print("best score {}".format(perf_score[best_idx]))
        print(performance_pd.loc[best_idx])

        if bidpriceest_model == 'linearBidPrice':
            best_base_bid = performance_pd['base_bid'][best_idx]
            best_pred_thresh = 0
        elif bidpriceest_model == 'linearBidPrice_variation':
            best_pred_thresh=performance_pd['pred_threshold'][best_idx]
            best_base_bid=performance_pd['base_bid'][best_idx]
        elif bidpriceest_model == 'linearBidPrice_mConfi':
            best_pred_thresh=performance_pd['pred_threshold'][best_idx]
            best_base_bid=performance_pd['base_bid'][best_idx]
            best_variable_bid = performance_pd['variable_bid'][best_idx]
        elif bidpriceest_model == 'thresholdsigmoid':
            #Return everything.
            best_pred_thresh=0
            best_base_bid=0
            pass

        # print("Pred_threshold\tbase_bid\tClick\tCTR\tSpend")
        # for _, i in performance_pd.iterrows():
        #     print("%.1f\t%d\t%d\t%.4f\t%.0f" % (i['pred_threshold'], i['base_bid'], i['click'], i['CTR'], i['spend']/1000))

        return best_pred_thresh,best_base_bid,performance_pd

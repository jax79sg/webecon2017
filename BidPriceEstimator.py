import numpy as np

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
        bids = [base_bid * (i/avg_ctr) for i in y_pred]
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
        bids = [base_bid + ((i-m_conf / 1-m_conf)*variable_bid) if i > m_conf else -1 for i in y_pred]
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


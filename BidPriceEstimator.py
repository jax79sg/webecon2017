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


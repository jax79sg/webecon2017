import numpy as np

class BidEstimator():

    def linearBidPrice(self, y_pred, base_bid, avg_ctr):
        p_ctr = np.count_nonzero(y_pred)/len(y_pred)
        bid = base_bid * (p_ctr/avg_ctr)

        print("p_ctr: %.3f\t avg_ctr: %.3f\t base_bid: %d\t bid: %d" %(p_ctr, avg_ctr, base_bid, bid))

        bidprice = np.full(len(y_pred), bid, dtype=int)
        return bidprice

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


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import     SMOTE, ADASYN

class ImbalanceSampling():
    "Other methods can be found in http://contrib.scikit-learn.org/imbalanced-learn/index.html"
    def createToyUnbalancedSamples(self, noOfLabels=2, noOfFeatures=20, noOfSamples=1000, unbalancedRatio=[0.1,0.9]):
        """
        Creates a toy version of samples (For testing purposes)
        :param noOfLabels:
        :param noOfFeatures:
        :param noOfSamples:
        :param unbalancedRatio:
        :return:
        """
        X, y = make_classification(n_classes=noOfLabels, class_sep=2,weights=unbalancedRatio, n_informative=3, n_redundant=1, flip_y=0,n_features=noOfFeatures, n_clusters_per_class=1, n_samples=noOfSamples, random_state=10)
        print('Created sample dataset label {}'.format(Counter(y)))
        return X,y

    def oversampling_SMOTE(self, X, y):
        print('Existing dataset label {}'.format(Counter(y)))
        os = SMOTE(random_state=42)
        X_res, y_res = os.fit_sample(X, y)
        print('Resampled dataset label {}'.format(Counter(y_res)))
        return X_res,y_res

    def oversampling_ADASYN(self, X, y):
        print('Existing dataset label {}'.format(Counter(y)))
        os = ADASYN()
        X_res, y_res = os.fit_sample(X, y)
        print('Resampled dataset label {}'.format(Counter(y_res)))
        return X_res,y_res


imbalanceSamp=ImbalanceSampling()
X,y=imbalanceSamp.createToyUnbalancedSamples()
X,y=imbalanceSamp.oversampling_SMOTE(X,y)
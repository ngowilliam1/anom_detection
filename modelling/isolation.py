import numpy as np
import time
from sklearn.ensemble import IsolationForest

class Isolation_Forest:
    def __init__(self, x_train=None, numpy_seed=123, max_samples=128, n_estimators=100, include_cat=True):
        rng = np.random.RandomState(numpy_seed)
        self.max_samples = max_samples
        self.include_cat = include_cat
        self.clf = IsolationForest(n_estimators=n_estimators,behaviour="new", max_samples=self.max_samples, random_state=rng, contamination='auto')
        if x_train is not None:
            self.fit(x_train)

    def fit(self, x):
        start = time.time()
        self.clf.fit(x)
        return (time.time()-start)

    def predict(self, x):
        return self.clf.predict(x)

    def score(self, x):
        # The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers. As described in the IsolationForest API
        # This is why we take the negative sign, as we want positive to represent that it is an outlier
        return (-self.clf.decision_function(x))
        
    def evaluate(self, x):
        return self.score(x)

    def evaluateV2(self,x):
        return -self.clf.score_samples(x)
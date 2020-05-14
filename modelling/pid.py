from modelling import pid_source
import numpy as np
class pidForest:
    def __init__(self, **kwargs):
        self.pidForest = pid_source.Forest(**kwargs)
    def fit(self, x):
        return self.pidForest.fit(np.transpose(x))
    def evaluate(self, x):
        _,_,_,_, score = self.pidForest.predict(np.transpose(x), err = 0.1, pct=50)
        # return (0-score) as score reflects the the likelihood that x is normal, but we want our evaluate to generate the reverse.
        return (0-score)
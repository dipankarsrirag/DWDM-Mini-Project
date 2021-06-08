import numpy as np

class NaiveBayes(object):

    def __init__(self, *, smoothing: float = 1.0):

        self.smoothing = smoothing
        self.priors = None
        self.features_proba = None
        self.is_fit = False

    def fit(self, X: np.array, y: np.array):
    
        split_classes = [[x for x, cat in zip(
            X, y) if cat == cls] for cls in np.unique(y)]
        self.priors = [np.log(len(x) / X.shape[0]) for x in split_classes]
        frequencies = np.array([np.array(cls).sum(axis=0)
                               for cls in split_classes]) + self.smoothing
        self.features_proba = np.log(
            frequencies / frequencies.sum(axis=1)[np.newaxis].T)
        self.is_fit = True
        return self

    def predict(self, X: np.array):
        
          assert self.is_fit, "The model must be fit before predicting!"
          return np.argmax([(self.features_proba * x).sum(axis=1) + self.priors for x in X], axis=1)

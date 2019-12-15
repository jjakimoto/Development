# Thrid party
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class Model:
    """ Lasso model with standardization

    Note:
        We consider predicting price. To neutrize output, we 
        set fit_intercept into False.
    """
    def __init__(self, **kwargs):
        self._model = Lasso(fit_intercept=False, **kwargs)
        self._pre = StandardScaler(with_mean=False)
        
    def fit(self, X, y):
        X = self._pre.fit_transform(X)
        self._model.fit(X, y)
        
    def predict_clf(self, X, th):
        pred = self.predict(X)
        output = np.zeros_like(pred)
        output[pred >= th] = 1
        output[pred <= -th] = -1
        return output
    
    def predict(self, X):
        X = self._pre.transform(X)
        pred = self._model.predict(X)
        return pred

import numpy as np

class LogisticRegression(object):

    def __init__(self, alpha=0.001, maxTraining=500):
        self.alpha = alpha
        self.maxTraining = maxTraining


    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(x))


    def _gradient_descend(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        theta = np.mat(np.zeros(len(y)).T)
        loop = 0
        while (loop < self.maxTraining):
            E = self._sigmoid(np.dot(X, theta)) - y
            theta = theta - self.alpha * np.dot(X.T, E)
            loop += 1
        return theta


    def fit(self, X, y):
        self.theta = self._gradient_descend(X, y)


    def predict(self, test):
        labels = []
        test = np.mat(test)
        for i in range(len(test)):
            pred_yi = self._sigmoid(np.dot(self.theta, test[i, :]))
            labels.append(1 if pred_yi >= 0.5 else 0)
        return labels






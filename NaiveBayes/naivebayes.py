import numpy as np

__author__ = 'Wenning Ding'


class NaiveBayes(object):
    '''
    Naive Bayes Classifier
    only fit with discrete features

    Parameters
    -----------
    alpha : float (default = 1.0)
            alpha = 1 : Laplace smoothing
            alpha = 0 : No smoothing
    '''


    def __init__(self, alpha=1.0):
        self.alpha = alpha


    # calculate conditional probability, input is a feature column
    def _cal_prob(self, feature):
        total = len(feature)
        value_num = {}
        for x in feature:
            if x not in value_num:
                value_num[x] = 1
            else:
                value_num[x] += 1
        for x in value_num:
            value_num[x] = (value_num[x] + self.alpha) / (total + self.alpha * len(value_num))

        return value_num


    def fit(self, X, y):

        # calculate P(Y = k)
        self.Y_prob = {}
        for yi in y:
            if not yi in self.Y_prob:
                self.Y_prob[yi] = 1
            else:
                self.Y_prob[yi] += 1
        for k in self.Y_prob:
            self.Y_prob[k] = self.Y_prob[k] + self.alpha / len(y) + self.alpha * len(self.Y_prob)
        # print(X[y==k])

        # calculate P(Xi = xi | Y = k)
        self.condition_prob = {}
        for k in self.Y_prob:
            self.condition_prob[k] = {}
            for i in range(len(X[0])):
                self.condition_prob[k][i] = self._cal_prob(X[y==k][:,i])
        print(self.condition_prob)


    # calculate posterior probability for one line
    def _pred(self, line):
        prob = []
        for k in self.Y_prob:
            prob_of_k = self.Y_prob.get(k)
            for i in range(len(line)):
                prob_of_k *= self.condition_prob.get(k).get(i).get(line[i])
            prob.append(prob_of_k)
        print(prob)
        index = prob.index(max(prob))
        return list(self.Y_prob.keys())[index]


    def predict(self, X):
        labels = []
        for i in range(len(X)):
            labels.append(self._pred(X[i,:]))
        return labels



class GaussianNaiveBayes(NaiveBayes):
    '''
    Gaussian Naive Bayes Classifier
    Fit with continuous features

    Parameters
    -----------
    alpha : float (default = 1.0)
            alpha = 1 : Laplace smoothing
            alpha = 0 : No smoothing
    '''


    # calculate conditional probability, input is a feature column
    def _cal_prob(self, feature):
        mean = np.mean(feature)
        std = np.std(feature)
        return (mean, std)


    # calculate posterior probability for one line
    def _pred(self, line):
        prob = []
        for k in self.Y_prob:
            prob_of_k = self.Y_prob.get(k)
            for i in range(len(line)):
                mean = self.condition_prob.get(k).get(i)[0]
                two_std_square = (self.condition_prob.get(k).get(i)[1]**2)*2     # 2*(std^2)
                prob_of_k *= np.exp(-(line[i] - mean)**2 / two_std_square) / np.sqrt(np.pi * two_std_square)
            prob.append(prob_of_k)
        index = prob.index(max(prob))
        return list(self.Y_prob.keys())[index]

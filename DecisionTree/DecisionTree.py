import numpy as np

__author__ = 'Wenning Ding'


class DecisionTree(object):
    '''
    Decision Tree Classifier

    Parameters
    -----------
    mode :  string (default = 'CART')
            mode = 'ID3'
            mode = 'C4.5'
    '''

    def __init__(self, mode='CART'):
        self.mode = mode
        self._tree = None


    # calculate Gini coefficient
    def _cal_Gini(self, label):
        classes = np.unique(label)
        total = float(len(label))
        res = 1.0
        for k in classes:
            res -= (len(label[label==k]) / total)**2
        return res


    # split data and label into two set
    def _split_data(self, X, y, feature, value):
        X1 = X[X[:, feature] >= value]
        X2 = X[X[:, feature] < value]
        y1 = y[X[:, feature] >= value]
        y2 = y[X[:, feature] < value]
        return X1, X2, y1, y2


    # vote for the max labels as label
    def _vote(self, y):
        res = np.unique[y]
        sum = [len(y[y==i]) for i in res]
        index = sum.index(max(sum))
        return res[index]


    def _create_tree(self, X, y, indexSet):
        # if all the labels in y are the same, return label
        if (len(np.unique(y)) == 1):
            return y[0, 0]

        # if there's no feature can be split, vote for the label
        if (len(indexSet) == 0):
            return self._vote(y)

        col_index, split_value = self._choose_colIndex_and_splitValue(X, y, indexSet)
        tree = {}
        tree['col_index'] = col_index
        tree['split_value'] = split_value

        indexSet.remove(col_index)
        X1, X2, y1, y2 = self._split_data(X, y, col_index, split_value)

        tree['left'] = self._create_tree(X1, y1, indexSet)
        tree['right'] = self._create_tree(X2, y2, indexSet)
        return tree


    # choose feature index and the value to split feature
    def _choose_colIndex_and_splitValue(self, X, y, indexSet):
        Gini_cur = self._cal_Gini(y)
        total = len(y)
        min_Gini = Gini_cur
        colIndex = None
        splitValue = None

        for i in indexSet:
            split_values = np.unique(X[:, i])
            for value in split_values:
                X1, X2, y1, y2 = self._split_data(X, y, i, value)
                if len(X1) == 0 or len(X2) == 0:
                    continue
                Gini = (len(y1) * self._cal_Gini(y1) + len(y2) * self._cal_Gini(y2)) / total

                if (Gini < min_Gini):
                    colIndex = i
                    splitValue = value
                    min_Gini = Gini

        return colIndex, splitValue


    def fit(self, X, y):
        y = y.T
        indexSet = set(i for i in range(len(X[0])))
        self._tree = self._create_tree(X, y, indexSet)
        print(self._tree)
        self._prune()



    def _pred(self, tree, line):
        if not isinstance(tree, dict):
            return tree

        if line[tree['col_index']] >= tree['split_value']:
            return self._pred(tree['left'], line)
        else:
            return self._pred(tree['right'], line)


    def predict(self, data):
        labels = []
        for i in range(len(data)):
            labels.append(self._pred(self._tree, data[i, :]))
        return labels


    def _prune(self):
        pass

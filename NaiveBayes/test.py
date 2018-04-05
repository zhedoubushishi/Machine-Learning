import numpy as np
from naivebayes import NaiveBayes, GaussianNaiveBayes

X = np.array([
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
])
X = X.T
y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

test = np.array([
    [2, 4],
    [3, 5]])



model1 = NaiveBayes(alpha=1.0)
model1.fit(X, y)
res1 = model1.predict(test)
print(res1)


model2 = GaussianNaiveBayes()
model2.fit(X, y)
res2 = model2.predict(test)
print(res2)

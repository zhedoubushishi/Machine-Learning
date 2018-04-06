from DecisionTree import DecisionTree
import numpy as np

# Toy data
X = np.array([[1, 2, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 0, 0, 1],
     [2, 1, 1, 0, 1],
     [1, 1, 0, 1, 1]])
y = np.array([['A', 'A', 'B', 'C', 'C']])

test = np.array([[1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 1]])

model = DecisionTree(mode='CART')
model.fit(X, y)

res = model.predict(test)
print(res)

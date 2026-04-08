import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

print("Test Prediction:", model.predict(X_test))

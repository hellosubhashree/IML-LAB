from sklearn.svm import SVR

X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

model = SVR(kernel='linear')
model.fit(X, y)

print(model.predict([[5]]))

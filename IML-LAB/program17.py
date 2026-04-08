from sklearn import svm

X = [[1, 2], [2, 3], [3, 3], [8, 7], [9, 8]]
y = [0, 0, 0, 1, 1]

model = svm.SVC(kernel='linear')
model.fit(X, y)

print(model.predict([[4, 4]]))

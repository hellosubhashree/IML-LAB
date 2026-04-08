import numpy as np

x1=np.array([3,1,2,4,3])

x2=np.array([6,3,4,7,5])

y=np.array([1,0,0,1,1])

x=np.column_stack((np.ones(5),x1,x2))

B=np.linalg.inv(x.T @ x) @ x.T @ y

b0,b1,b2=B

print("intercept(bo): ",b0)
print("coefficient for studied hours(b1): ",b1)
print("coefficient for sleep hours(b2): ",b2)



import math
def sigmoid(x):
  return 1 / (1+math.exp(-x))

#step 1: calculate the linear combination (z=b0 + b1*x2)
z= x @ B
print("\nlinear combination(z):")
print(z)

#step 2: apply sigmoid to get probabilities
y_prob=np.array([sigmoid(val)for val in z])
print("\npredicted prbabilities (sigmoid output): ")
print(y_prob)

y_pred = (y_prob>= 0.5).astype(int)
print("\npredited classes: ")
print(y_pred)

TP=np.sum((y == 1) & (y_pred == 1))
TN=np.sum((y == 0) & (y_pred == 0))
FP=np.sum((y == 0) & (y_pred == 1))
FN=np.sum((y == 1) & (y_pred == 0))
print("\n confusion matrix: ")
print(f"TP = {TP},FP={FP}")
print(f"FN = {FN},TN={TN}")

#accuracy
accuracy= (TP + TN) / len(y)
#prcision
precision = TP/(TP + FP) if (TP + FP) != 0 else 0
#recall
recall= TP/(TP + FN) if (TP + FN) != 0 else 0
#f1_score
f1_score = 2*(precision * recall) / (precision * recall)

print("/nclassification matrix: ")
print(f"accuracy = {accuracy: 4f}")
print(f"precison = {precision: 4f}")
print(f"recall = {recall: 4f}")
print(f"F1 score = {f1_score: 4f}")
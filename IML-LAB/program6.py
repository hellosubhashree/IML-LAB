import numpy as np
from sklearn import svm
x=np.array([2,2],
           [4,5],
           [7,4])
y=np.array([-1,1,1])
clf=svm.SVC(kernel='linear',C=1000)
clf.fit(x,y)
weights = clf.coef_[0]
intercept=clf.intercept_[0]
support_vectors=clf.support_vectors_
print("--- svm numerical results ---")
print(f"weight(w):{weights}")

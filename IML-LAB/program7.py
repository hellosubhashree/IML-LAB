import numpy as np

x=np.array([1,2,3])
y1=np.array([2,4,5])
y2=np.array([10,20,25])

x=np.column_stack((np.ones(3),x))

y=np.column_stack((y1,y2))

B=np.linalg.inv(x.T @ x) @ x.T @ y

b0_y1,b0_y2=B[0]
b1_y1,b1_y2=B[1]

print("y1: intercept=", b0_y1,"slope",b1_y1)
print("y2: intercept=", b0_y2,"slope",b1_y2)
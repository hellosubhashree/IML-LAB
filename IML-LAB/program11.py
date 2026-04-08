import numpy as np

x=np.array([1,2,3])
y=np.array([2,4,5])

x=np.column_stack((np.ones(3),x))

B=np.linalg.inv(x.T @ x) @ x.T @ y

b0,b1=B

print("intercept(bo)",b0)
print("coefficient for x1(b1)",b1)

import numpy as np

x1=np.array([1,2,1])
x2=np.array([2,2,1])
y=np.array([6,9,4])

x=np.column_stack((np.ones(3),x1,x2))

B=np.linalg.inv(x.T @ x) @ x.T @ y

b0,b1,b2=B

print("intercept(bo)",b0)
print("coefficient for studied hours(b1)",b1)
print("coefficient for sleep hours(b2)",b2)
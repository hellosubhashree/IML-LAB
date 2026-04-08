import numpy as np

x1=np.array([1,2])
y=np.array([1,3])

x=np.column_stack((np.ones(2),x1))

b=np.linalg.inv(x.T @ x) @ x.T @y

b0,b1=b

print("intercept(bo)",b0)
print("coefficient for study hour(b1)",b1)
##print("coefficient for sleep hour(b2)",b2)

y_pred=b0+b1*x1
print("predicted y:",y_pred)

for lam in[0,1,5,100]:
 I = np.eye(x.shape[1])
 I[0,0]=0
 b=np.linalg.inv(x.T @ x+lam *I*I) @ x.T @y
 print(f"lambda={lam}, coefficients={b}")




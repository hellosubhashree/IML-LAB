import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

x=np.array([
    [1,2,10],
    [2,1,12],
    [3,3,9],
    [4,5,11],
    [5,4,13],
    [6,6,8],
    [7,7,7],
    [8,8,14]
])
y=np.array([6,7,12,18,20,24,28,32])

df=pd.DataFrame(x,columns=["x1","x2","x3"])
df["y"]=y
print("Dataset: \n",df)

lr=LinearRegression()
lr.fit(x,y)
y_pred_lr=lr.predict(x)

print("\nLinear Regression(no Regularization)")
print("coeffcient: ",lr.coef_)
print("MSE:",mean_squared_error(y,y_pred_lr))

ridge=Ridge(alpha=1.0)
ridge.fit(x,y)
y_pred_ridge = ridge.predict(x)

print("\nRidge Regression(L2 Regularization)")
print("coeffcients: ",ridge.coef_)
print("MSE:",mean_squared_error(y,y_pred_ridge))

lasso=Lasso(alpha=0.5)
lasso.fit(x,y)
y_pred_lasso = lasso.predict(x)

print("\nLasso Regression(L2 Regularization)")
print("coeffcients: ",lasso.coef_)
print("MSE:",mean_squared_error(y,y_pred_lasso))

import numpy as np

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x,y)
y_pred_lr=lr.predict(x)

ridge=Ridge(alpha=1.0)
ridge.fit(x,y)
y_pred_ridge = ridge.predict(x)
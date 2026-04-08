#import required librarires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#given data
probability_score=np.array([.9,.8,.7,.6,.55,.54,.53,.51,.50,.40])
classes=['p','p','n','p','p','n','n','n','p','n']

#convert class level to binary
y_true=np.array([1 if c=='p' else 0 for c in classes])

#calculate roc_curve
fpr, tpr,thresholds=roc_curve(y_true, probability_score)

#calculate AUC
roc_auc=auc(fpr,tpr)

#plot roc_curve
plt.figure(figsize=(7,6))
plt.plot(fpr,tpr,color='green',lw=2, label='roc curve(AUC =%0.3f)'%roc_auc)
plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--',label='random classifier')

plt.xlim([0.0,1.0])
plt.xlim([0.0,1.05])
plt.xlabel('false positive rate(FPR)')
plt.xlabel('true positive rate(TPR)')
plt.title('RECIEVER OPERATING CHARACTERISTICS (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#pirnt AUC
print("AUc value: ",roc_auc)


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from SVM import SVM

data=datasets.load_iris()
X=data.data
Y=data.target
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
print(X_train,X_test)
plt.figure()
plt.scatter(X[:,2],X[:,3],c=Y,s=40)
plt.show()
model=SVM()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
som=0
x=0
for i,j in enumerate(pred):
    som=som+1
    if pred[i]==Y_test[i]:
        x=x+1
print(x/som*100)
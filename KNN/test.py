import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN

database=datasets.load_iris()
X=database.data
Y=database.target
print(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
plt.figure()
plt.scatter(X[:,2],X[:,3],c=Y,s=40)
plt.show()
model=KNN(5)
model.fit(X_train,Y_train)
pred=model.predict(X_test)  
# envoie les prédictions pour chaque point 
# n the context of a KNN model, model.predict(X_test) would make predictions on the test data X_test using the fitted KNN model model. The method _predict(x) is called on each sample in X_test to compute the predicted class label, based on the k nearest neighbors in the training data. The predicted class labels are returned as an array pred.
som=0
x=0
for i,j in enumerate(pred):
    som=som+1
    if pred[i]==Y_test[i]:
        x=x+1
print(x/som*100)
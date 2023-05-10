import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import data
path = "../data/wine-data/winequality-red.csv"
df = pd.read_csv(path, sep = ';')

#replace quality with binary class
df["class"] = [1 if i >= 5 else 0 for i in df["quality"]]
df = df.drop(columns = ["quality"])

#split data into features and target
y = df['class']
features = df.drop(columns = ['class'])

#transform data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features)

#split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)


##Logistic Regression without Regularization##


#define classifier without regularization
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty=None, max_iter=1000) #penalty=None means no regularization
clf.fit(X_train, y_train)

#plot coefficients
coef = clf.coef_.ravel() #ravel() flattens the array
predictors = features.columns

coef = pd.Series(coef, predictors).sort_values()
coef.plot(kind = 'bar', title = 'Model Coefficients without Regularization')
# plt.show()

#evaluate model using f1 score
from sklearn.metrics import f1_score
y_pred_train = clf.predict(X_train) #predict on train data
y_pred_test = clf.predict(X_test) #predict on test data
print("F1 score on training data: ", f1_score(y_train, y_pred_train))
print("F1 score on testing data: ", f1_score(y_test, y_pred_test))


##L2 Regularization##


#train model with L2 regularization
clf_l2 = LogisticRegression(max_iter=1000) #penalty='l2' is default
clf_l2.fit(X_train, y_train)

#evaluate model using f1 score
y_pred_train_l2 = clf_l2.predict(X_train) #predict on train data
y_pred_test_l2 = clf_l2.predict(X_test) #predict on test data
print("F1 score on training data l2: ", f1_score(y_train, y_pred_train_l2))
print("F1 score on testing data l2: ", f1_score(y_test, y_pred_test_l2))

#observe f1 score remains the same for regularized model and non-regularized model
#this is because the the constraint boundary for the regularization we performed is 
#large enough to hold the original loss function minimum
#thus rendering our model the same as the unregularized one.


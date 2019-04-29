#this a model created using sklearn open library
import pandas as pd 
import numpy as np 
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings       

train = pd.read_csv('/home/user/Desktop/tensorflow/loan_prediction/train.csv')
test = pd.read_csv('/home/user/Desktop/tensorflow/loan_prediction/test.csv')
train_original = train.copy() #copy of train data, to preserve original data from changes
test_original = train.copy()


#treating missing values & Outliers

print(train.isnull().sum())

#missing data and outliers can have adverse effect on the model performance. Fill numerical values with mean and catogarical values with mode.
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

print(train['Loan_Amount_Term'].value_counts()) #loan amount term 360 is the mode 

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

#use median to fill the null values in loan amount. as earlier we saw that loan amount have outliers so the mean will not be the 
#proper approach as it is highly affected by the presence of outliers
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#treat missing values in test data as we did for training data
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#remove the skewness in loan amount by doing the log transformation. it does not affect the smaller values much, but reduces the larger values.
train['LoanAmount_log'] = np.log(train['LoanAmount']) 
test['LoanAmount_log'] = np.log(test['LoanAmount'])


#===================================== Model Training =================================================

#no need Loan_ID for training so drop it
train = train.drop('Loan_ID',axis=1) 
test = test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1) 
y = train.Loan_Status

#this will convert catogarical data to 0 and 1 which makes easy to learn by the model
X = pd.get_dummies(X) 
train = pd.get_dummies(train) 
test = pd.get_dummies(test)


i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1,shuffle = True)
for train_index, test_index in kf.split(X,y):
	print('\n{} of kfold {}'.format(i,kf.n_splits))
	xtr,xvl = X.loc[train_index],X.loc[test_index]     
	ytr,yvl = y[train_index],y[test_index]

	model = LogisticRegression(random_state=1, solver = 'lbfgs', max_iter = 10000)
	model.fit(xtr, ytr)  
	pred_test = model.predict(xvl)
	score = accuracy_score(yvl,pred_test)
	print('accuracy_score',score)
	i += 1

pred_test = model.predict(test)
pred = model.predict_proba(xvl)[:,1]


submission = pd.read_csv("sample_submission.csv")
submission['Loan_Status'] = pred_test 
submission['Loan_ID'] = test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')



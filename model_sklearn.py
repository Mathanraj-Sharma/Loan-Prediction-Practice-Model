#this a model created using sklearn open library
import pandas as pd 
import numpy as np 
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
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

#split train data into train and validation
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

model = LogisticRegression(solver='lbfgs') 
model.fit(x_train, y_train)

#set hyperparameters 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, 
	max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=1, solver='liblinear', tol=0.0001,          
	verbose=0, warm_start=False)

#validate trained model using validation set 
pred_cv = model.predict(x_cv)
score = accuracy_score(y_cv,pred_cv)

print("Model Accuracy = ",score)

#prediction
pred_test = model.predict(test)

submission = pd.read_csv("sample_submission.csv")
submission['Loan_Status'] = pred_test 
submission['Loan_ID'] = test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')



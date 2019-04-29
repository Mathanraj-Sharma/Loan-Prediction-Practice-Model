import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")

train = pd.read_csv('/home/user/Desktop/tensorflow/loan_prediction/train.csv')
test = pd.read_csv('/home/user/Desktop/tensorflow/loan_prediction/test.csv')

train_original = train.copy() #copy of train data, to preserve original data from changes
test_original = train.copy()

print(train.columns)#check the columns
print(train.dtypes) #check datatypes
print(train.shape, test.shape) #check the number of rows and columns

#=========> Univarient Analysis

print("Loan_Status counts: \n",train['Loan_Status'].value_counts()) #Catogarical counts of Loan_Status
print("Loan_Status counts: \n",train['Loan_Status'].value_counts(normalize = True)) #Catogarical propotions of Loan_Status

train['Loan_Status'].value_counts().plot.bar()#plot Loan_Status

plt.figure(2)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 

plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.figure(3)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

plt.subplot(132) 
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 

plt.subplot(133) 
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 

plt.figure(4)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))

plt.figure(5)
plt.subplot(121) 
sns.distplot(train['CoapplicantIncome'])

plt.subplot(122) 
train['CoapplicantIncome'].plot.box(figsize=(16,5)) 

plt.figure(6)
plt.subplot(121) 
df=train.dropna() 
sns.distplot(df['LoanAmount'])

plt.subplot(122) 
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()



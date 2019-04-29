import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")

train = pd.read_csv('/home/user/Desktop/tensorflow/loan_prediction/train.csv')
test = pd.read_csv('/home/user/Desktop/tensorflow/loan_prediction/test.csv')

train_original = train.copy() #copy of train data, to preserve original data from changes
test_original = train.copy()

 #plot relationship between catogarical data and loan status
Gender = pd.crosstab(train['Gender'],train['Loan_Status']) 
Married = pd.crosstab(train['Married'],train['Loan_Status']) 
Dependents = pd.crosstab(train['Dependents'],train['Loan_Status']) 
Education = pd.crosstab(train['Education'],train['Loan_Status']) 
Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status']) 
Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status']) 
Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status']) 



Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
  
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
 
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
 
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

#mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

#create bins for the applicant income variable based on the values in it and analyze the corresponding loan status for each bin

bins = [0,2500,4000,6000,81000] #0-2499, 2500-3999, 4000-5999, 6000-80999
groupA = ['Low','Average','High', 'Very high'] 
train['Income_bin'] = pd.cut(train['ApplicantIncome'],bins,labels = groupA)

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True) 
plt.xlabel('ApplicantIncome') 
plt.ylabel('Percentage')

#coapplicant income and loan amount variable 
bins = [0,1000,3000,42000] 
groupC = ['Low','Average','High'] 
train['Coapplicant_Income_bin'] = pd.cut(train['CoapplicantIncome'],bins,labels = groupC)

Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage') 

"""It shows that if coapplicant’s income is less the chances of loan approval are high. 
But this does not look right. The possible reason behind this may be that most of the 
applicants don’t have any coapplicant so the coapplicant income for such applicants is 0 
and hence the loan approval is not dependent on it. So we can make a new variable in which 
we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval."""

train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
bins = [0,2500,4000,6000,81000] 
groupT = ['Low','Average','High', 'Very high']
train['Total_Income_bin'] = pd.cut(train['Total_Income'],bins,labels = groupT)

Total_Income_bin = pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
plt.xlabel('Total_Income') 
plt.ylabel('Percentage') 

#visualize the Loan amount variable
bins = [0,100,200,700] 
groupL = ['Low','Average','High'] 
train['LoanAmount_bin'] = pd.cut(train['LoanAmount'], bins, labels = groupL)

LoanAmount_bin = pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount') 
plt.ylabel('Percentage')

plt.show()

#drop the bins which we created for the exploration part
train = train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

#change the 3+ in dependents variable to 3 to make it a numerical variable
train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True)

#convert the target variable’s categories into 0 and 1 so that we can find its correlation with numerical variables.  replace N with 0 and Y with 1.
train['Loan_Status'].replace('N', 0,inplace=True) 
train['Loan_Status'].replace('Y', 1,inplace=True)

#create heat map to visualize correlation between all the numerical variables
matrix = train.corr() 
ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")
plt.show()
#most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History - Loan_Status). LoanAmount is also correlated with CoapplicantIncome
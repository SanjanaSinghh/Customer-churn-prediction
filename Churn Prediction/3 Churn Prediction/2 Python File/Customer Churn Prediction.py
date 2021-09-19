#!/usr/bin/env python
# coding: utf-8

# 
# **Customer Churn Prediction:**
# 
#     A Bank wants to take care of customer retention for its product: savings accounts. The bank wants you to identify customers likely to churn balances below the minimum balance. You have the customers information such as age, gender, demographics along with their transactions with the bank.
# 
#     Your task as a data scientist would be to predict the propensity to churn for each customer.
# 
# **Data Dictionary**
#     
#     There are multiple variables in the dataset which can be cleanly divided into 3 categories:
#     
#     I. Demographic information about customers
#     
#     customer_id - Customer id 
#     
#     vintage - Vintage of the customer with the bank in a number of days 
#     
#     age - Age of customer 
#     
#     gender - Gender of customer 
# 
#     dependents - Number of dependents 
# 
#     occupation - Occupation of the customer 
# 
#     city - City of the customer (anonymized) 
# 
#     II. Customer Bank Relationship
# 
#     customer_nw_category - Net worth of customer (3: Low 2: Medium 1: High) 
# 
#     branch_code - Branch Code for a customer account 
# 
#     days_since_last_transaction - No of Days Since Last Credit in Last 1 year 
# 
#     III. Transactional Information
# 
#     current_balance - Balance as of today 
# 
#     previous_month_end_balance - End of Month Balance of previous month 
# 
#     average_monthly_balance_prevQ - Average monthly balances (AMB) in Previous Quarter 
# 
#     average_monthly_balance_prevQ2 - Average monthly balances (AMB) in previous to the previous quarter 
# 
#     current_month_credit - Total Credit Amount current month 
# 
#     previous_month_credit - Total Credit Amount previous month 
# 
#     current_month_debit - Total Debit Amount current month 
# 
#     previous_month_debit - Total Debit Amount previous month 
# 
#     current_month_balance - Average Balance of current month 
# 
#     previous_month_balance - Average Balance of previous month 
# 
#     churn - Average balance of customer falls below minimum balance in the next quarter (1/0) 
#  
# 
#     (Note: In the same downloaded folder, you can find the dataset (churn_prediction) for this problem statement. Once you upload the final project, you will be able to download the project solution (Final project solution.zip). This folder contains the Jupyter notebook file that contains the solution to this final project problem statement.)
# 
# 
# 

# # 01 Importing the Liabraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install seaborn')


# In[2]:


data =  pd.read_csv("churn_prediction.csv")
data.head()


# # 02 Data Preprocessing (EDA and Data Manipulation):-

# ### 2.1 Exploratory Data Analysis (EDA):-

# In[3]:


# Checking all the datatypes
data.dtypes


# In[4]:


# CHecking the Shape of the DataFrame
data.shape


# In[5]:


# Checking for Null Values

data.isnull().sum()


# In[6]:


# Univariate Analysis

data.columns


# In[7]:


temp = ['customer_id', 'vintage', 'age', 'gender', 'dependents', 'occupation',
       'city', 'customer_nw_category', 'branch_code',
       'days_since_last_transaction', 'current_balance',
       'previous_month_end_balance', 'average_monthly_balance_prevQ',
       'average_monthly_balance_prevQ2', 'current_month_credit',
       'previous_month_credit', 'current_month_debit', 'previous_month_debit',
       'current_month_balance', 'previous_month_balance', 'churn']

for i in temp:
    if data[i].dtypes == 'object':
        print("UNIVARIATE ANALYSIS FOR " + i)
        plt.figure(figsize = (10,5))
        data[i].value_counts().plot.bar()
        #plt.xlabel(i)
        plt.show()


# In[8]:


for i in temp:
    if data[i].dtypes != 'object':
        print("UNIVARIATE ANALYSIS FOR " + i)
        plt.figure(figsize = (10,5))
        data[i].plot.hist(bins = 50)
        #plt.xlabel(i)
        plt.show()
        
        


# # Bivariate Analysis:-

# In[9]:


for i in temp:
    if data[i].dtypes!= 'object':
        plt.figure(figsize = (10,5))
        plt.plot(data[i],data["churn"])
        plt.xlabel(i)
        plt.ylabel("Customer Churn")
        plt.title(i + " VS Customer Churn")


# In[10]:


data.select_dtypes('object').head()


# In[11]:


data.select_dtypes(['int','float']).head()


# In[12]:


#data.isnull().sum()
data.isnull().sum()


# In[13]:


data['gender'].mode()


# In[14]:


data['occupation'].mode()


# In[15]:


# Imputing Catagorical Var:
data['gender']=data['gender'].fillna('Male')
data['occupation']=data['occupation'].fillna('self_employed')


# In[16]:


# Imputing Continious Var:
data['dependents']= data['dependents'].fillna(data['dependents'].mean())
data['city']= data['city'].fillna(data['city'].mean())
data['days_since_last_transaction']= data['days_since_last_transaction'].fillna(data['days_since_last_transaction'].mean())


# In[17]:


#data.isnull().sum()
data.isnull().sum()


# # Univariate Analysis (After Tuning) :-

# In[18]:


temp = ['customer_id', 'vintage', 'age', 'gender', 'dependents', 'occupation',
       'city', 'customer_nw_category', 'branch_code',
       'days_since_last_transaction', 'current_balance',
       'previous_month_end_balance', 'average_monthly_balance_prevQ',
       'average_monthly_balance_prevQ2', 'current_month_credit',
       'previous_month_credit', 'current_month_debit', 'previous_month_debit',
       'current_month_balance', 'previous_month_balance', 'churn']

for i in temp:
    if data[i].dtypes == 'object':
        print("UNIVARIATE ANALYSIS FOR " + i)
        plt.figure(figsize = (10,5))
        data[i].value_counts().plot.bar()
        #plt.xlabel(i)
        plt.show()


# In[19]:


for i in temp:
    if data[i].dtypes != 'object':
        print("UNIVARIATE ANALYSIS FOR " + i)
        plt.figure(figsize = (10,5))
        data[i].plot.hist(bins = 50)
        #plt.xlabel(i)
        plt.show()


# In[20]:


data_new = data.copy()


# In[21]:


data_new.head()


# In[22]:


data_new.isnull().sum()


# # Checking for Value Counts:-
# 

# In[23]:


data_new.columns


# In[24]:


temp_01 = ['customer_id', 'vintage', 'age', 'gender', 'dependents', 'occupation',
       'city', 'customer_nw_category', 'branch_code',
       'days_since_last_transaction', 'current_balance',
       'previous_month_end_balance', 'average_monthly_balance_prevQ',
       'average_monthly_balance_prevQ2', 'current_month_credit',
       'previous_month_credit', 'current_month_debit', 'previous_month_debit',
       'current_month_balance', 'previous_month_balance', 'churn']

for i in temp_01:
    print(" Value Counts for ", i)
    print(data[i].value_counts())


# In[25]:


# Value Counts for gender and occupation

data["gender"].value_counts()


# In[26]:


data["occupation"].value_counts()


# In[27]:


data["gender"]= data["gender"].map({"Male":1,"Female":0})


# In[28]:


data.head()


# In[29]:


pd.get_dummies(data["occupation"]).head()


# In[30]:


data.head()


# In[31]:


y = pd.get_dummies(data["occupation"])


# In[32]:


y.columns


# In[33]:


data.columns


# In[34]:


data.head()


# In[35]:


data.drop(["occupation"], axis = 1, inplace = True)


# In[36]:


data.head()


# In[37]:


data.columns


# In[38]:


for i in range(len(y.columns)):
    data.insert(5+i,y.columns[i],y[y.columns[i]])


# In[39]:


data.head()


# In[40]:


data.describe()


# In[41]:


data.head()


# In[42]:


data.columns


# # Bivariate Analysis (After Tuning):-

# In[44]:


temp2 = ['customer_id', 'vintage', 'age', 'gender', 'dependents', 'company',
       'retired', 'salaried', 'self_employed', 'student', 'city',
       'customer_nw_category', 'branch_code', 'days_since_last_transaction',
       'current_balance', 'previous_month_end_balance',
       'average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2',
       'current_month_credit', 'previous_month_credit', 'current_month_debit',
       'previous_month_debit', 'current_month_balance',
       'previous_month_balance', 'churn']
for i in temp2:
    if data[i].dtypes!= 'object':
        plt.figure(figsize = (10,5))
        plt.plot(data[i],data["churn"])
        plt.xlabel(i)
        plt.ylabel("Customer Churn")
        plt.title(i + " VS Customer Churn")


# # Modellng:-

# In[45]:


# Separating Input and Output Variables:-

x = data.drop(["churn"],axis= 1)
y = data["churn"]


# In[46]:


x.head()


# In[47]:


y.head()


# # Standerdization :-

# In[48]:


from sklearn.preprocessing import StandardScaler


# In[49]:


scaler = StandardScaler()


# In[50]:


# Standerising inputs:-
x = scaler.fit_transform(x)


# In[51]:


print(x)


# # Train_Test_Split

# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


x_train, x_test, y_train , y_test =  train_test_split(x,y, random_state= 20, stratify =  y , test_size = 0.2)


# In[54]:


x_train


# In[55]:


x_test


# In[56]:


y_train


# In[57]:


y_test


# # Calling Classifiers :- 

# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[59]:


new_scaler = StandardScaler()
new_scaler.fit_transform(x_train,x_test)


# In[60]:


# Fitting into models
classifier1 = LogisticRegression()
classifier1.fit(x_train,y_train)

classifier2 = KNeighborsClassifier()
classifier2.fit(x_train,y_train)

classifier3 = RandomForestClassifier()
classifier3.fit(x_train,y_train)


# In[76]:


y_pred1 = classifier1.predict(x_test)


# In[77]:


y_pred2 = classifier2.predict(x_test)


# In[78]:


y_pred3 = classifier3.predict(x_test)


# In[80]:


y_pred1, y_pred2, y_pred3 


# # Checking Accuracy

# In[65]:


# Imp[orting Metrics
from sklearn.metrics import classification_report ,confusion_matrix,accuracy_score

print("***************************************************************************************************************")
print("For Logistic Regression")
results1 = confusion_matrix(y_test, y_pred1)
print('Accuracy Score :',accuracy_score(y_test, y_pred1))
print( classification_report(y_test, y_pred1))
print("***************************************************************************************************************")

print("For KNN Classifier")
results2 = confusion_matrix(y_test, y_pred2)
print('Accuracy Score :',accuracy_score(y_test, y_pred2))
print( classification_report(y_test, y_pred2))
print("***************************************************************************************************************")


print("For Random Forest")
results3 = confusion_matrix(y_test, y_pred3)
print('Accuracy Score :',accuracy_score(y_test, y_pred3))
print( classification_report(y_test, y_pred3))
print("***************************************************************************************************************")


# # Visualization

# In[66]:


plt.figure(figsize = (10,6))
plt.plot(y_test,y_pred1)
plt.xlabel("Actual Churn")
plt.ylabel("Predicted Churn using Logistic Regression")
plt.title("Actual Churn VS Predicted (Logistic Regression) Churn")


# In[67]:


plt.figure(figsize = (10,6))
plt.plot(y_test,y_pred2)
plt.xlabel("Actual Churn")
plt.ylabel("Predicted Churn using KNN")
plt.title("Actual Churn VS Predicted (KNN) Churn")


# In[68]:


plt.figure(figsize = (10,6))
plt.plot(y_test,y_pred3)
plt.xlabel("Actual Churn")
plt.ylabel("Predicted Churn using Random Forest")
plt.title("Actual Churn VS Predicted (Random Forest) Churn")


# # Model Ensembling:-

# In[69]:


from sklearn.ensemble import VotingClassifier

#classifier1 = LogisticRegression()
#classifier1.fit(x_train,y_train)

#classifier2 = KNeighborsClassifier()
#classifier2.fit(x_train,y_train)

#classifier3 = RandomForestClassifier()
#classifier3.fit(x_train,y_train)

evc = VotingClassifier(estimators = [
                                    ("Logistic Regression",classifier1), 
                                    ("KNeighborsClassifier",classifier2), 
                                    ("RandomForestClassifier",classifier3)]
                                    ,voting = 'hard')


# In[70]:


evc.fit(x_train,y_train)


# In[71]:


LR_SCORE = classifier1.score(x_test,y_test)
KNN_SCORE  = classifier2.score(x_test,y_test)
RANDOM_FOREST_SCORE = classifier3.score(x_test,y_test)
ENSEMBLE_SCORE = evc.score(x_test,y_test)


# In[72]:


# Creating a DataFrame to show results:-

my_dict = {"LR_SCORE":[LR_SCORE],
           "KNN_SCORE":[KNN_SCORE],
           "RANDOM_FOREST_SCORE":[RANDOM_FOREST_SCORE],
           "ENSEMBLE_SCORE":[ENSEMBLE_SCORE]
          }


# In[73]:


Final_Results = pd.DataFrame(my_dict)
Final_Results.index.names = [None]


# In[74]:


Final_Results


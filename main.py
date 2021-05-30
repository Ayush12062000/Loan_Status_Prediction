#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder 
    from sklearn.metrics import accuracy_score,confusion_matrix
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import warnings
    import time
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys , os
    from sklearn import metrics
    print("modules imported")

except:
    print("module not found")

DeprecationWarning("ignore")
warnings.filterwarnings("ignore")


# In[2]:


os.listdir()


# In[3]:


try:
    df=pd.read_csv('credit_risk.csv')
    print("data imported")
except:
    print("data not imported")


# In[4]:


df.columns


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.Dependents.value_counts()


# In[10]:


sns.distplot(np.where(df['Loan_Status']=='Y',1,0), hist = True, kde= False)


# In[11]:


'''From this graph i can say that person getting loan is twice as compare to not getting Loan'''


# In[12]:


mode_gender=df.Gender.mode()

mode_married=df.Married.mode()

mode_depend=df.Dependents.mode()

mode_self=df.Self_Employed.mode()

mean_loanamount=df.LoanAmount.mean()

mean_loan_term=df.Loan_Amount_Term.mean()

mean_credit=df.Credit_History.mean()


# In[13]:


#Filling Null data and Using Label encoder
def fill_gender(df):
    df.Gender.fillna('Male', inplace=True)
    return df

def fill_depend(df):
    df.Dependents.fillna('0', inplace= True)
    return df

def fill_married(df):
    df.Married.fillna('Yes' ,inplace= True)
    return df

def fill_self(df):
    df.Self_Employed.fillna('No',inplace=True)
    return df

def fill_loanamount(df):
    df.LoanAmount.fillna(mean_loanamount,inplace=True)
    return df

def fill_loanterm(df):
    df.Loan_Amount_Term.fillna(mean_loan_term,inplace=True)
    return df

def fill_credit(df):
    df.Credit_History.fillna(mean_credit,inplace=True)
    return df

def label_encode(df):
    from sklearn.preprocessing import LabelEncoder 
    label=LabelEncoder()
    df.Gender=label.fit_transform(df['Gender'])
    df.Education = label.fit_transform(df['Education'])
    df.Married=label.fit_transform(df['Married'])
    df.Dependents=label.fit_transform(df['Dependents'])
    df.Self_Employed=label.fit_transform(df['Self_Employed'])
    df.Loan_Status=label.fit_transform(df['Loan_Status'])
    return df

df=fill_gender(df)
df=fill_depend(df)
df=fill_married(df)
df=fill_loanamount(df)
df=fill_self(df)
df=fill_loanterm(df)
df=fill_credit(df)
df=label_encode(df)


# In[14]:


df.isnull().sum()


# In[15]:


df.info()


# In[16]:


#spliting data in train test
train,test =train_test_split(df,test_size=0.2,random_state=13)


# In[17]:


cat = train.select_dtypes(include = object)
for col in cat:
    sns.countplot(train[col].dropna())
    plt.show()


# In[18]:


sns.countplot(train.Married) # majority is of yes, infact its double


# In[19]:


#Dividing into X and Y
def x_and_y(df):
    x=df.drop(["Loan_ID","Property_Area","Married","Loan_Status","Gender"],axis=1)
    y=df["Loan_Status"]
    return x, y

x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)


# In[20]:


x_train.info()


# In[21]:


#Training Our Model
#Decision tree
d_model= DecisionTreeClassifier()
d_model.fit(x_train, y_train)


# In[22]:


print("Decision tree Train score:{}".format(d_model.score(x_train,y_train)))
print("Decision tree Test score:{}".format(d_model.score(x_test,y_test)))


# In[23]:


#confusion matrix
from sklearn.metrics import confusion_matrix
x_test_prediction=d_model.predict(x_test)
confusion_matrix(y_test, x_test_prediction)
pd.crosstab(y_test, x_test_prediction, rownames = ['Actual'], colnames =['Predicted'], margins = True)


# In[24]:



from sklearn.metrics import recall_score, precision_score
recall=recall_score(y_test, x_test_prediction)
precision=precision_score(y_test, x_test_prediction)
print("recall = {} \nprecision = {}".format(recall,precision))


# In[27]:


#Precision is 0.88235 or, when it predicts that a aplicant will get Loan, it is correct around 88% of the time.


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test, x_test_prediction))


# In[ ]:





# In[ ]:





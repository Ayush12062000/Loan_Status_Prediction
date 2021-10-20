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
    df=pd.read_csv('Loan.csv')
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


df.Married.mode()


# In[13]:


#Filling Null data and Using Label encoder
from sklearn.preprocessing import LabelEncoder

def preprocessing(df):
    
    df.Gender.fillna('Male', inplace=True)
    df.Dependents.fillna('0' , inplace= True)
    df.Married.fillna('Yes' ,inplace= True)
    df.Self_Employed.fillna('No',inplace=True)
    df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)
    df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(),inplace=True)
    df.Credit_History.fillna(df.Credit_History.mean(),inplace=True)
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

#%%
df = preprocessing(df) 
df = label_encode(df)


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
    x=df.drop(["Loan_ID","Property_Area","Married","Loan_Status","Gender","Education"],axis=1)
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


# In[38]:


#Precision is 0.8809 or, when it predicts that an applicant will get Loan, it is correct around 88% of the time.


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test, x_test_prediction))


# # Save Model

# In[33]:


import pickle

with open("classifier.bin", "wb") as fout:
    pickle.dump(d_model, fout)
    fout.close()


# In[34]:


#Loading model

with open("classifier.bin","rb") as fin:
    model = pickle.load(fin)
    fin.close()


# In[35]:


os.listdir()


# In[46]:


testdata = pd.read_csv("test.csv")


# In[47]:


testdata


# In[49]:


testdata = preprocessing(testdata)


# In[50]:


def test_label_encode(df):
    from sklearn.preprocessing import LabelEncoder 
    label=LabelEncoder()
    df.Gender=label.fit_transform(df['Gender'])
    df.Education = label.fit_transform(df['Education'])
    df.Married=label.fit_transform(df['Married'])
    df.Dependents=label.fit_transform(df['Dependents'])
    df.Self_Employed=label.fit_transform(df['Self_Employed'])
    return df

testdata = test_label_encode(testdata)


# In[51]:


testdata.head()


# In[52]:


testdata.isnull().sum()


# In[54]:


testdata = testdata.drop(["Loan_ID","Property_Area","Married","Gender","Education"],axis=1)


# In[55]:


testdata.columns


# In[56]:


x_train.columns


# In[57]:


prediction = model.predict(testdata)


# In[58]:


prediction


# In[ ]:
'''x = [[0,0,5720,0,110.0,360.0,1.000000],[3,1,450000,177879,2000000,2458976,1.000000]]

#%%
x = pd.DataFrame(x, columns= ['Dependents', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History'])
#%%
x = x.to_json()'''

#%%
#model.predict(x)

# %%
import requests

#url = 'http://127.0.0.1:5000/'
url = 'https://prediction-loan-status.herokuapp.com/'

X = {"Dependents":[2,3],"Self_Employed":[0,1],"ApplicantIncome":[3881,450000],"CoapplicantIncome":[0,177879],"LoanAmount":[147.0,2000000.0],"Loan_Amount_Term":[360.0,78459.0],"Credit_History":[0.0,1.0]}

result = requests.post(url,json=X)
result.text.strip()

# %%
# 0 means Loan Declined, 1 means Loan approved
#%%
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

#%%
#Path of the file (CSV)
os.chdir('E:/Machine_learning/Models/Credit_risk')
os.listdir()

#%%
#Reading data from CSV file
try:
    df=pd.read_csv('credit_risk.csv')
    print("data imported")
except:
    print("data not imported")

#%%
os.getcwd()

#%%
df.describe()

#%%
df.info()   #tells all the details about data

#%%
df.Dependents.value_counts()

sns.distplot(np.where(df['Loan_Status']=='Y',1,0), hist = True, kde= False)
'''From this graph i can say that person getting loan are double to not approved'''

#%%
sns.countplot(train.Married) # majority is of yes, infact its double

#%%
cat = train.select_dtypes(include = object)
for col in cat:
    sns.countplot(train[col].dropna())
    plt.show()

#%%
mode_gender=df.Gender.mode()

mode_married=df.Married.mode()

mode_depend=df.Dependents.mode()

mode_self=df.Self_Employed.mode()

mean_loanamount=df.LoanAmount.mean()

mean_loan_term=df.Loan_Amount_Term.mean()

mean_credit=df.Credit_History.mean()


#%%
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


#%%
df.isnull().sum()

#%%
#spliting data in train test
train,test =train_test_split(df,test_size=0.2,random_state=13)

#%%
#Dividing into X and Y
def x_and_y(df):
    x=df.drop(["Loan_ID","Married","Property_Area","Loan_Status","Education","Gender","ApplicantIncome"],axis=1)
    y=df["Loan_Status"]
    return x, y

x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)

#%%
#Training Our MAchine
#Decision tree
d_model= DecisionTreeClassifier(criterion= 'entropy',max_depth= 5,max_features=4,random_state=13,min_samples_leaf=2)
d_model.fit(x_train, y_train)
print("Decision tree Train score:{}".format(d_model.score(x_train,y_train)))
print("Decision tree Test score:{}".format(d_model.score(x_test,y_test)))

'''
#KNN
from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
print("KNN Train score:{}".format(clf.score(x_train,y_train)))
print("KNN Test score:{}".format(clf.score(x_test,y_test)))'''

#%%
#confusion matrix
from sklearn.metrics import confusion_matrix
x_test_prediction=d_model.predict(x_test)
confusion_matrix(y_test, x_test_prediction)

#%%
from sklearn.metrics import recall_score, precision_score
recall=recall_score(y_test, x_test_prediction)
precision=precision_score(y_test, x_test_prediction)
print("recall ={} \n precision = {}".format(recall,precision))

#%%
from sklearn.model_selection import cross_val_score
cv=cross_val_score(d_model,x_train,y_train,cv=5)
print(cv)
print(cv.mean())
print(cv.std())

#%%
'''
from sklearn.svm import LinearSVC
s=LinearSVC(C=0.1)
s.fit(x_train,y_train)

print("SVM Train score:{}".format(s.score(x_train,y_train)))
print("SVM Test score:{}".format(s.score(x_test,y_test)))'''


# %%
'''
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)

print("LOGISTIC Train score:{}".format(lg.score(x_train,y_train)))
print("LOGISTIC Test score:{}".format(lg.score(x_test,y_test)))'''


# %%

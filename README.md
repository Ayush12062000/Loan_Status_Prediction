Here , I Have the code for the Credit Risk dataset . In this dataset there were many details like ...

Loan_ID              614 non-null object
Gender               614 non-null int32
Married              614 non-null int32
Dependents           614 non-null int32
Education            614 non-null object
Self_Employed        614 non-null int32
ApplicantIncome      614 non-null int64
CoapplicantIncome    614 non-null float64
LoanAmount           614 non-null float64
Loan_Amount_Term     614 non-null float64
Credit_History       614 non-null float64
Property_Area        614 non-null object
Loan_Status          614 non-null int32

In this we have to predict whether the person will get loan(1) or not(0).
''' Loan id is not useful
Loan_status is target variable'''
After Cleaning the data (Filling Null data, converting String into integer) , I have used Various Classification Algorithms.(COZ it's Classification Problem)
Decision Tree Gave better Results.


And Uploading CSV file as well.......


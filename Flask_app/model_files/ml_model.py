import pandas as pd
import numpy as np
def label_encode(df):
    from sklearn.preprocessing import LabelEncoder 
    label=LabelEncoder()
    df.Dependents=label.fit_transform(df['Dependents'])
    df.Self_Employed=label.fit_transform(df['Self_Employed'])
    return df

def predict_loan(config, model):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    df = label_encode(df)
    y_pred = model.predict(df)
    z = y_pred.astype(np.float)
    return z
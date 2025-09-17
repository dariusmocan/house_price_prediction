import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

df = pd.read_csv("../data/boston.csv")
print(df.head())  

X = df.drop(["TAX"],axis=1)
y = np.log(df["TAX"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("MSE : ",mean_squared_error(y_test,y_pred))
print("r2_score : ", r2_score(y_test,y_pred))

# print(df[["TAX"]].describe())





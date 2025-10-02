import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

df = pd.read_csv("../data/house.csv")
# print(df.head())  

#basic filtering
df = df[df["Price"].between(100000, 800000)]
df["Garage"] = df["Garage"].map({"No" : 0, "Yes" : 1})

#feature engineering
df["HouseAge"] = datetime.now().year - df["YearBuilt"]
df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']


# one-hot encoding
df = pd.get_dummies(df,columns=["Location","Condition"], drop_first=True)
# print(df.head())

X = df.drop(["Id","Price"],axis=1)
# X = df.drop(["Id","Price","Location", "Condition", "Garage"], axis = 1)
y = np.log(df["Price"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("MSE : ",mean_squared_error(y_test,y_pred))
print("r2_score : ", r2_score(y_test,y_pred))

print(df[["Price","Area", "Bedrooms", "Bathrooms"]].describe())





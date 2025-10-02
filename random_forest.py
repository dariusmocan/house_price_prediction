import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

df = pd.read_csv("../data/house.csv")

df = df[df["Price"].between(100000, 800000)]
df["Garage"] = df["Garage"].map({"Yes" : 1, "No" : 0}) 

#feature engineering
df["HouseAge"] = datetime.now().year - df["YearBuilt"]
df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']

#one-hot encoding
df = pd.get_dummies(df,columns=["Location","Condition"], drop_first=True)
# print(df.head())

X = df.drop(["Price","Id"],axis=1)
y = np.log(df["Price"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf = RandomForestRegressor(n_estimators=500,n_jobs=-1,random_state=42)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print("MSE : ", mean_squared_error(y_test,y_pred))
print("r2_score : ", r2_score(y_test,y_pred))


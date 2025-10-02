import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

df = pd.read_csv("../data/boston.csv")
print(df.head())  

X = df.drop(["MEDV"],axis=1)
y = np.log(df["MEDV"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("MSE : ",mean_squared_error(y_test,y_pred))
print("r2_score : ", r2_score(y_test,y_pred))

# print(df[["TAX"]].describe())

# hardcoded house pred
new_house = {
    "CRIM": 0.1,        # Crime rate
    "ZN": 18.0,         # Proportion of residential land zoned for large lots
    "INDUS": 2.31,      # Proportion of non-retail business acres
    "CHAS": 0,          # Charles River dummy (1 if tract bounds river; 0 otherwise)
    "NOX": 0.538,       # Nitrogen oxides concentration
    "RM": 6.5,          # Average number of rooms per dwelling
    "AGE": 65.2,        # Proportion of owner-occupied units built prior to 1940
    "DIS": 4.09,        # Weighted distances to employment centers
    "RAD": 1,           # Index of accessibility to radial highways
    "TAX": 296,         # Property tax rate per $10,000 (caracteristică cunoscută)
    "PTRATIO": 15.3,    # Pupil-teacher ratio
    "B": 396.9,         # Proportion of blacks by town
    "LSTAT": 4.98       # % lower status of the population
}

house_df = pd.DataFrame([new_house])

house_df = house_df[X.columns]

log_price_pred = lr.predict(house_df)[0]
price_pred = np.exp(log_price_pred)

print("New House:")
for feature, value in new_house.items():
    print(f"  {feature}: {value}")

print(f"\nPrice Prediction: ${price_pred:.2f}k")


# user input house pred
print("\n" + "="*50)
print("INSERT NEW HOUSE DATA")
print("="*50)

user_house = {}
for feature in X.columns:
    while True:
        try:
            value = float(input(f"INSERT {feature}: "))
            user_house[feature] = value
            break
        except ValueError:
            print("Insert a valid number")

user_house_df = pd.DataFrame([user_house])
user_house_df = user_house_df[X.columns]

user_log_price_pred = lr.predict(user_house_df)[0]
user_price_pred = np.exp(user_log_price_pred)

# print("\n Custom house inserted by user")
# for feature, value in user_house.items():
#     print(f"  {feature}: {value}")

print(f"\nPrice Prediction: ${user_price_pred:.2f}k")

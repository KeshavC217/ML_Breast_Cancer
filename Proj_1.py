import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def score_RFR(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


def score_DTR(X_train, X_val, y_train, y_val):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


dataset = pd.read_csv(r'C:\Users\kesha\Documents\data.csv')

# 1 for Malignant and 0 for benign
maligno = {'M': 1 ,'B': 0}
dataset.diagnosis = [maligno[item] for item in dataset.diagnosis]

y = dataset.diagnosis
features = ['radius_mean','texture_mean', 'area_mean', 'perimeter_mean', 'smoothness_mean','concavity_mean','concave points_mean', 'symmetry_mean']
X = dataset[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_val = pd.DataFrame(imputer.transform(X_val))

# Although this particular dataset might not have any missing values, it is good practice to use an imputer, in case any
# expansions include missing values
imputed_X_train.columns = X_train.columns
imputed_X_val.columns = X_val.columns

# Applying a one hot coding to deal with categorical variables:

a = (X_train.dtypes == 'object')
object_cols = list(a[a].index)

# Using a one hot encoder approach
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

ohct = pd.DataFrame(ohe.fit_transform(X_train[object_cols]))
ohcv = pd.DataFrame(ohe.transform(X_val[object_cols]))

# One-hot encoding removed index; put it back
ohct.index = X_train.index
ohcv.index = X_val.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_val.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OHE_X_train = pd.concat([num_X_train, ohct], axis=1)
OHE_X_valid = pd.concat([num_X_valid, ohcv], axis=1)




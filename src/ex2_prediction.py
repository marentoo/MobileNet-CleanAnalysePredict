#Linear Regression

"""""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score


# Load data
data = pd.read_csv('downloads.csv')

# Define features and target
features = ['chipsettime', 'mcsindex', 'tbs0', 'tbs1', 'rb0', 'rb1']
target = 'throughput'

# Convert target to binary
threshold = 50
data['target_binary'] = np.where(data[target] > threshold, 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['target_binary'], test_size=0.2, random_state=42)

# create a logistic regression model
lr = LinearRegression()

# fit the model on the training data
lr.fit(X_train, y_train)

# make predictions on the test data
y_pred = lr.predict(X_test)

# convert probabilities to binary
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# compute mean squared error
mse = mean_squared_error(y_test, y_pred_binary)
print("Mean Squared Error:", mse)

# compute root mean squared error
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# compute R-squared
r2 = r2_score(y_test, y_pred_binary)
print("R-squared:", r2)

# Compute F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score:", f1)

# Compute Precision score
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

# Compute Recall score
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)


plt.scatter(y_test, y_pred_binary)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
"""
"""""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('downloads.csv')

# Define features and target
features = ['chipsettime', 'mcsindex', 'tbs0', 'tbs1', 'rb0', 'rb1']
target = 'throughput'

# Convert target to binary
threshold = 50
data['target_binary'] = np.where(data[target] > threshold, 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['target_binary'], test_size=0.2, random_state=42)

# create a logistic regression model
lr = LinearRegression()

# fit the model on the training data
lr.fit(X_train, y_train)

# make predictions on the test data
y_pred = lr.predict(X_test)

# convert probabilities to binary
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Compute F1 score
f1 = f1_score(y_test, y_pred_binary, average='weighted')
print("F1 score:", f1)

# Compute Precision score
precision = precision_score(y_test, y_pred_binary, average='binary', zero_division=1)
print("Precision:", precision)

# Compute Recall score
recall = recall_score(y_test, y_pred_binary, average='binary', zero_division=1)
print("Recall:", recall)

# Create heatmap for precision and recall
sns.heatmap(data=cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import confusion_matrix

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(cm)

"""""

#Decision Trees
#works
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Load the data from CSV file
data = pd.read_csv("downloads.csv")

data = pd.get_dummies(data, columns=['mobiProv_name', 'mcs0', 'mcs1']) #get the string values

# Split the data into training and testing sets
X = data.drop(['throughput'], axis=1)
y = data['throughput']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Mean Squared Error (MSE)")
print("Root Mean Squared Error (RMSE)")
''''
print('---------------Decision Trees------------------')

# create a decision tree model
dt = DecisionTreeRegressor(random_state=42)

# fit the model on the training data
dt.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dt.predict(X_test)

# compute mean squared error
mse = mean_squared_error(y_test, y_pred)
# compute root mean squared error
rmse = mse ** 0.5
# compute R-squared
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)

print('---------------Linear Regression------------------')

# create a logistic regression model
lr = LinearRegression()

# fit the model on the training data
lr.fit(X_train, y_train)

# make predictions on the test data
y_pred = lr.predict(X_test)

# compute mean squared error
mse = mean_squared_error(y_test, y_test)
# compute root mean squared error
rmse = np.sqrt(mse) # type: ignore
# compute R-squared
r2 = r2_score(y_test, y_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
'''
'''''
print('---------------Random Forest------------------') #taking long, so be patient

# create a random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# fit the model on the training data
rf.fit(X_train, y_train)

# make predictions on the testing data
y_pred = rf.predict(X_test)

# compute mean squared error
mse = mean_squared_error(y_test, y_test)
# compute root mean squared error
rmse = np.sqrt(mse) # type: ignore
# compute R-squared
r2 = r2_score(y_test, y_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)

print('---------------Support Vector Machines------------------') #taking long, so be patient

# create a support vector machines model
svm = SVR(kernel="linear") #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’

# fit the model on the training data
svm.fit(X_train, y_train)

# make predictions on the testing data
y_pred = svm.predict(X_test)

# compute mean squared error
mse = mean_squared_error(y_test, y_test)
# compute root mean squared error
rmse = np.sqrt(mse) # type: ignore
# compute R-squared
r2 = r2_score(y_test, y_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
'''

print('---------------Gradient Boosting------------------') 

# create a gradient boosting model
gb = GradientBoostingRegressor()

# fit the model on the training data
gb.fit(X_train, y_train)

# make predictions on the testing data
y_pred = gb.predict(X_test)

# compute mean squared error
mse = mean_squared_error(y_test, y_test)
# compute root mean squared error
rmse = np.sqrt(mse) # type: ignore
# compute R-squared
r2 = r2_score(y_test, y_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
# import numpy as np
# from sklearn.linear_model import LinearRegression


# #model 1 - Linear Regression

# # Define features and target

# def linear_regression_predict(data):



#     features = ['chipsettime', 'mcsindex', 'tbs0', 'tbs1', 'rb0', 'rb1']
#     target = 'throughput'
#     # Convert target to binary
#     threshold = 50
#     data['target_binary'] = np.where(data[target] > threshold, 1, 0)
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(data[features], data['target_binary'], test_size=0.2, random_state=42)
#     # create a logistic regression model
#     lr = LinearRegression()
#     # fit the model on the training data
#     lr.fit(X_train, y_train)
#     # make predictions on the test data
#     y_pred = lr.predict(X_test)
#     # convert probabilities to binary
#     y_pred_binary = np.where(y_pred > 0.5, 1, 0)
#     # Compute confusion matrix
#     cm = confusion_matrix(y_test, y_pred_binary)
#     # Compute F1 score
#     f1 = f1_score(y_test, y_pred_binary, average='weighted')
#     print("F1 score:", f1)
#     # Compute Precision score
#     precision = precision_score(y_test, y_pred_binary, average='binary', zero_division=1)
#     print("Precision:", precision)
#     # Compute Recall score
#     recall = recall_score(y_test, y_pred_binary, average='binary', zero_division=1)
#     print("Recall:", recall)
#     # Create heatmap for precision and recall
#     sns.heatmap(data=cm, annot=True, cmap="Blues")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.show()
#     from sklearn.metrics import confusion_matrix
#     # calculate confusion matrix
#     cm = confusion_matrix(y_test, y_pred_binary)
#     print("Confusion Matrix:")
#     print(cm)



# #model 2 - Decision tree



# #model 3 - SVM




# #model 4 - Random forest



# #model 5 - Gradient boosting





# # def predict(df1, df2):
#     # linear_regression_predict(df_downloads)
#     # linear_regression_predict(df_uploads)
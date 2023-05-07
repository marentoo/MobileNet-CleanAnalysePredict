""" *Note
On the bottom there is function predict() - chose function you want to call for regression or classification - depending on target - predicted value
then chose which model - parameter
uncoment what you want to initialize and comment other functions """

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os

#------------------------------------------------------------------------------------
models_reg= (
    DecisionTreeRegressor(random_state = 42),
    #SVR(kernel="linear"), #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’) Dlugo?!
    GradientBoostingRegressor(),
    RandomForestRegressor(n_estimators=60, random_state=42),
    LinearRegression()
    )

def prediction_regression(model, data, columns_feature, target, dataset_name):
        directory = 'evaluation'
        if not os.path.exists(directory):
                os.makedirs(directory)
        file_path_ev = os.path.join(directory, f'{type(model).__name__}_plots_{dataset_name}.png')   
                
                #load data
        data = pd.get_dummies(data, columns = columns_feature)
        X = data.drop([target], axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

                #build model
        model = model
        # fit the model on the training data
        model.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = model.predict(X_test)

                # evaluate
        # compute mean squared error
        mse = mean_squared_error(y_test, y_pred)
        # compute root mean squared error
        rmse = mse ** 0.5
        # compute R-squared
        r2 = r2_score(y_test, y_pred)

        print('---------------',type(model).__name__,'------------------')
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R-squared:", r2)

        #plot the scores
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        fig.savefig(file_path_ev)


#-------------------------------------------------------------------------------------------------
models_clas = (DecisionTreeClassifier(random_state=42),
               GradientBoostingClassifier(),
               RandomForestClassifier(),
               SVC(kernel="linear") #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’

)

def prediction_classification(model, data, columns_feature, target, dataset_name):
        directory = 'evaluation'
        if not os.path.exists(directory):
                os.makedirs(directory)
        file_path_ev2 = os.path.join(directory, f'{type(model).__name__}_ConfusionMatrix_{dataset_name}.png')
                # Load data
        data = pd.get_dummies(data, columns=columns_feature)
        X = data.drop([target], axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create a decision tree model
        model = model
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)
        
        encoder = LabelBinarizer()
        y_test_one_hot = encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))

                #Evaluate
        #Compute accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print('---------------',type(model).__name__,'------------------')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)
        
        #Plot the confusion matrix
#...
        plt.figure(figsize=(10,8))
        sns.set(font_scale = 1.4)
        sns.heatmap(cm, annot=True, annot_kws={"size":16}, cmap = 'Blues',fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(ticks=[0,1,2], labels=['o2', 'telekom', 'vodafone'])
        plt.yticks(ticks=[0,1,2], labels=['o2', 'telekom', 'vodafone'])
        plt.savefig(file_path_ev2)

        #plol ROC curve
#...
        n_classes = len(np.unique(y_test))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range (n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
                roc_auc[i]= auc(fpr[i], tpr[i])

        plt.figure(figsize=(8,6))
        colors = ['blue', 'red', 'green']  # specify colors for each class
        for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # plot diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curves')
        plt.legend(loc="lower right")
        plt.savefig(f'evaluation/{type(model).__name__}_ROC_curve_{dataset_name}.png')

        # precision recall curve ...


#-------------------------------------------------------------------------------------------------
def predict(downloads, uploads, merged):

        prediction_regression(models_reg[0], downloads, ['mobiProv_name', 'mcs0', 'mcs1'],'throughput','downloads')
        prediction_regression(models_reg[0], uploads, ['mobiProv_name'],'tp_cleaned' ,'uploads')
        prediction_regression(models_reg[0], merged,['mcs', 'mobiProv_name' , 'proces_DorU'],'throughput','merged')

        # prediction_classification(models_clas[0], downloads, ['mcs0', 'mcs1'],'mobiProv_name','downloads')
        # prediction_classification(models_clas[0], uploads, [],'mobiProv_name' ,'uploads')
        # prediction_classification(models_clas[0], merged,['mcs',  'proces_DorU'],'mobiProv_name','merged')

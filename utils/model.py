import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


def visualize_model(X_train : pd.DataFrame, y_train : pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,inputs_columns : list[str],model):
    # train random forest model
    model.fit(X_train[inputs_columns], y_train)

    # predict on test set
    y_pred = model.predict(X_test[inputs_columns])

    # # accuracy score
    print(accuracy_score(y_test, y_pred))

    # confusion matrix and f1 score
    print("Confusion matrix of primary model : \n" , confusion_matrix(y_test, X_test['side'].replace(-1,0)))
    print("Confusion matrix of secondary model : \n" , confusion_matrix(y_test, y_pred))
    print("F1 score of primary model : \n" , classification_report(y_test, X_test['side'].replace(-1,0)))
    print("F1 score of secondary model : \n" , classification_report(y_test, y_pred))

    # feature importance
    print(model.feature_importances_)

    y_pred_proba = model.predict_proba(X_test[inputs_columns])
    return_series = pd.Series(y_pred_proba[:,1]*X_test['return']*X_test['side'])

    print(return_series.describe())

    # plot return series
    plt.title('Return series of meta label model')
    return_series.cumsum().plot()
    plt.show()


def weighted_sampling(sampling_indices:np.ndarray,distribution:np.ndarray,num_samples:int) -> list[int]:
    # normalize distribution
    distribution = distribution / np.sum(distribution)
    # select sample index with replacement until num_samples is reached
    bag = []
    while (len(bag) <= num_samples):
        # select using numpy then record into bag
        bag.append(np.random.choice(sampling_indices,p=distribution))
    return bag

import math 
def bagging(X_train : pd.DataFrame, y_train : pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,inputs_columns : list[str],num_models:int,distribution:pd.Series,parameters:dict) -> np.ndarray:
    if num_models == None:
        num_models = math.ceil(1 / distribution.mean())
        print(f'Number of models is set to {num_models}')
    # create model list
    model_list = []
    y_pred_proba = np.zeros((num_models,len(X_test)),dtype=float)
    # for each model, sample from training data
    for i in range(num_models):
        # sample from training data
        selected_indices = weighted_sampling(X_train.index.values,distribution.values,int(distribution.mean()*X_train.shape[0]))
        X_train_sample = X_train.loc[selected_indices]
        y_train_sample = y_train.loc[selected_indices]

        model_list.append(RandomForestClassifier(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'], random_state=0))

        # train model
        model_list[i].fit(X_train_sample[inputs_columns], y_train_sample)
        
        # predict_proba with each model
        y_pred_proba[i] = model_list[i].predict_proba(X_test[inputs_columns])[:,1]
    
    
    # take average
    y_pred_proba = np.mean(y_pred_proba,axis=0)

    # return prediction
    return y_pred_proba

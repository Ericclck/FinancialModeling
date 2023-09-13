from utils.custom_class import get_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from utils.custom_class import *
import matplotlib.pyplot as plt

def MDI(estimator,df,sampling_cols,plot=True):
    print("Performing MDI, make sure preprocessing preserve all columns.")
    estimator.fit(df)
    
    out = [tree.feature_importances_ for tree in estimator.model.estimators_]

    out = pd.DataFrame(out,index=range(estimator.model.n_estimators),columns=sampling_cols)
    if plot:
        means = out.mean(axis=0)
        errors = out.std(axis=0)*(len(out)**-0.5)
        plt.figure(figsize=(10, 5))
        plt.title("MDI")
        plt.barh(means.index, means.values, xerr=errors.values, align='center', alpha=0.6, ecolor='black', capsize=10)
        plt.axvline(x=1/len(sampling_cols), color='r', linestyle='--')
        plt.show()
    return out

def MDA(estimator,df,cv,sampling_cols,plot=True):
    estimator.fit(df)
    # lock first_touch_time for all estimators
    cv.first_touch_time = estimator.labeler.first_touch_time.copy(deep=True)
    cv.first_touch_time = cv.first_touch_time.reindex(df.index,method="bfill").fillna(df.index[-1])
    # lock sampling dates for all estimators
    estimator.sampling_dates = df.iloc[estimator.sampling_indices].index
    out = pd.DataFrame(index=range(cv.num_combinations),columns=[c for c in sampling_cols])
    for i,(train_indices,test_indices) in enumerate(cv.split(df)):
        estimator.fit(df.iloc[train_indices])
        base_score = get_score(estimator,df.iloc[test_indices],True,0.001)
        for col in sampling_cols:
            estimator.feature_engineer.sampler.shuffle = col
            estimator.fit(df.iloc[train_indices])
            score = get_score(estimator,df.iloc[test_indices],True,0.001)
            out.loc[i,col] = (score-base_score)/base_score
            estimator.feature_engineer.sampler.shuffle = None

    if plot:
        means = out.mean(axis=0)
        errors = out.std(axis=0)*(len(out)**-0.5)
        plt.figure(figsize=(10, 5))
        plt.title("MDA (Score Improvement after shuffling)")
        plt.barh(means.index, means.values, xerr=errors.values, align='center', alpha=0.6, ecolor='black', capsize=10)
        plt.show()
    return out
        

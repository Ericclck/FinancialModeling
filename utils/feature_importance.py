from utils.custom_class import get_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def MDA(estimator,df,cv):
    estimator.fit(df)
    # lock first_touch_time for all estimators
    cv.first_touch_time = estimator.labeler.first_touch_time.copy(deep=True)
    cv.first_touch_time = cv.first_touch_time.reindex(df.index,method="bfill").fillna(df.index[-1])
    # lock sampling dates for all estimators
    estimator.sampling_dates = df[estimator.sampling_indices].index
    out = pd.DataFrame(index=range(cv.n_splits),columns=df.columns)
    for i,(train_indices,test_indices) in enumerate(cv.split(df)):
        estimator.fit(df.iloc[train_indices])
        prob = estimator.predict_proba(df.iloc[test_indices])
        # record meta_labels for shuffled columns
        estimator.meta_labels = estimator.labeler.meta_labels
        estimator.test_meta_labels = estimator.test_labeler.meta_labels
        # record true labels for shuffled columns
        true_labels = estimator.test_labeler.meta_labels
        base_score = -log_loss(true_labels,prob)
        for col in df.columns:
            df_copy = df.copy()
            np.random.shuffle(df_copy[col].values)
            estimator.fit(df_copy.iloc[train_indices])
            prob = estimator.predict_proba(df_copy.iloc[test_indices])
            score = -log_loss(true_labels,prob)
            out.loc[i,col] = (score-base_score)/base_score
        # reset meta_labels
        estimator.meta_labels = None
        estimator.test_meta_labels = None
    return out
        

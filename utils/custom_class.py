import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from utils.labeling import get_events,get_labels,ewma_std_pct_change
from utils.bars import get_events as get_cusum_events
from utils.bars import sampling
from pandas import Timestamp, DateOffset
from utils.sample_weights import *

class ClassifierWrapper(BaseEstimator,ClassifierMixin):
    def __init__(self,cusum_threshold,X_pipe,ptsl_scalers,model,min_target,ewma_window,primary_model,sample_weight,num_days_exit,sampling_dates=None) -> None:
        self.cusum_threshold = cusum_threshold
        self.X_pipe = X_pipe
        self.ptsl_scalers = ptsl_scalers
        self.model = model
        self.min_target = min_target
        self.ewma_window = ewma_window
        self.sampling_dates = sampling_dates
        self.primary_model = primary_model
        self.sample_weight = sample_weight
        self.num_days_exit = num_days_exit
        self.meta_labels = None
        self.test_meta_labels = None
    def fit(self,df,y=None):
        # print("Fit over date : ", df.index[0], " to ", df.index[-1])
        close = df.close.copy(deep=True)
        # Signal generation
        if self.sampling_dates is None:
            print("Warning : Only use this when first or not performing cross validation")
            self.sampling_indices = get_cusum_events(close, self.cusum_threshold)
        else:
            if not isinstance(self.sampling_dates,pd.Index) or not isinstance(self.sampling_dates,pd.DatetimeIndex):
                raise Exception("sampling_dates should be a pandas Index or a pandas DatetimeIndex")
            self.sampling_indices = close.index.searchsorted(self.sampling_dates[self.sampling_dates.isin(close.index)])
        
        # labeling
        self.labeler = Labeler(close,self.sampling_indices,self.ptsl_scalers,self.min_target,self.ewma_window,self.primary_model,self.num_days_exit)
        self.labeler.fit()
        if self.meta_labels is not None:
            print("Warning : meta_labels is not None, performing MDA training.")
            self.labeler.meta_labels = self.meta_labels
        y = self.labeler.meta_labels

        # feature engineering
        # print("Samples wasted by labeling",len(sampling(df,self.sampling_indices))-len(sampling(df,self.sampling_indices).loc[y.index]))
        self.sampled_df = sampling(df,self.sampling_indices).loc[y.index]
        self.X_preprocessed = self.X_pipe.fit_transform(self.sampled_df)

        # modeling
        if self.sample_weight:
            self.model.fit(self.X_preprocessed,y,sample_weight=get_sample_weights(self.labeler.first_touch_time,close))
        else:
            self.model.fit(self.X_preprocessed,y)
        return self
    def preprocessed_before_predict(self,df):
        # print("Predict over date : ", df.index[0], " to ", df.index[-1])
        close = df.close.copy(deep=True)

        # Signal generation
        self.test_sampling_indices = close.index.searchsorted(self.sampling_dates[self.sampling_dates.isin(close.index)])
        # labeling
        self.test_labeler = Labeler(close,self.test_sampling_indices,self.ptsl_scalers,self.min_target,self.ewma_window,self.primary_model,self.num_days_exit)
        self.test_labeler.fit()
        if self.test_meta_labels is not None:
            print("Warning : meta_labels is not None, performing MDA prediction.")
            self.test_labeler.meta_labels = self.test_meta_labels
        y = self.test_labeler.meta_labels

        # feature engineering
        return self.X_pipe.transform(sampling(df,self.test_sampling_indices).loc[y.index])
    def predict(self,df):
        if self.sampling_dates is None:
            self.test_labeler = self.labeler
            X = self.X_preprocessed
        else:
            X = self.preprocessed_before_predict(df)
        return self.model.predict(X)
    def predict_proba(self,df,verbose=False):
        if hasattr(self.model,'predict_proba'):
            if self.sampling_dates is None:
                self.test_labeler = self.labeler
                X = self.X_preprocessed
            else:
                X = self.preprocessed_before_predict(df)
            if verbose:
                print("Warning : Actual Prediction is performed, sampling_dates will be updated!")
                self.sampling_dates = df[self.sampling_indices].index
                last_sampling_date = self.sampling_dates[-1]
                if self.labeler.target[last_sampling_date] < self.min_target:
                    print("Warning : Target is below minimum target, This bet should not be taken")
                pt_level = self.ptsl_scalers[0] * self.labeler.target[last_sampling_date]
                sl_level = -self.ptsl_scalers[1] * self.labeler.target[last_sampling_date]
                print(f"(target,min_target) : ({self.labeler.target[-1]:.4f},{self.min_target}) , Should trade? : ",self.labeler.target[last_sampling_date] > self.min_target)
                print("Last sampling Date : ",last_sampling_date)
                print(f"Last sampling side, starting price: {self.labeler.side[last_sampling_date]},{df.close[last_sampling_date]:.2f}")
                print(f"Last sampling Profit, price , inflated price : ,{pt_level},{df.close[last_sampling_date]*(1+self.labeler.side[last_sampling_date]*pt_level):.2f},{df.close[last_sampling_date]*(1+self.labeler.side[last_sampling_date]*(pt_level*1.5)):.2f}")
                print(f"Last sampling Stop : ,{sl_level},{df.close[last_sampling_date]*(1-self.labeler.side[last_sampling_date]*abs(sl_level)):.2f}")
                print("Last sampling should stop at closing of : ",pd.to_datetime(np.busday_offset(last_sampling_date.strftime('%Y-%m-%d'), self.num_days_exit, roll='forward')))
            return self.model.predict_proba(X)[:,1]
        else:
            raise Exception('Model does not have predict_proba method')


class OutlierSmoother(BaseEstimator,TransformerMixin):
    def __init__(self,boundaries:list[float]) -> None:
        self.boundaries = boundaries
    def fit(self,X,y=None):
        self.low = np.percentile(X,self.boundaries[0],axis=0)
        self.high = np.percentile(X,self.boundaries[1],axis=0)
        return self
    def transform(self,X):
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                if X[row,col] < self.low[col]:
                    X[row,col] = self.low[col]
                elif X[row,col] > self.high[col]:
                    X[row,col] = self.high[col]
        return X
    
class ToDf(BaseEstimator,TransformerMixin):
    def __init__(self,columns:list[str]=None,index:pd.Index=None) -> None:
        self.columns = columns
        self.index = index
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        if self.columns == None: self.columns = [f'feature_{i}' for i in range(X.shape[1])]
        return pd.DataFrame(X,columns=self.columns,index=self.index)
    
class Featurizer(BaseEstimator,TransformerMixin):
    def __init__(self,transformer,discard_source=True) -> None:
        self.transformer = transformer
        self.discard_source = discard_source
    def fit(self,X,y=None):
        self.transformer.fit(X)
        return self
    def transform(self,X):
        if self.discard_source:
            return self.transformer.transform(X)
        else:
            return np.concatenate([X,self.transformer.transform(X)],axis=1)
        
        
class crossing_ma:
    def __init__(self,fast_slow:tuple[int,int]) -> None:
        self.fast_slow = fast_slow
    def predict(self,close,sampling_indices):
        fast,slow = self.fast_slow
        return pd.Series(np.where(close[sampling_indices].rolling(fast).mean()>close[sampling_indices].rolling(slow).mean(),1,-1),index=close[sampling_indices].index)
    def __repr__(self):
        return f"crossing_ma(fast_slow={self.fast_slow})"
class RSI:
    def __init__(self,rsi_period,rsi=None) -> None:
        self.rsi_period = rsi_period
        self.rsi = rsi
    def predict(self,close,sampling_indices):
        def calculate_rsi(data, window):
            delta = data.diff()
            up, down = delta.copy(), delta.copy()
            
            up[up < 0] = 0
            down[down > 0] = 0

            average_gain = up.rolling(window).mean()
            average_loss = abs(down.rolling(window).mean())

            rs = average_gain / average_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        if self.rsi:
            rsi = self.rsi[sampling_indices]
        else:
            rsi = calculate_rsi(close[sampling_indices],self.rsi_period)
        return pd.Series(
            np.where(rsi<50,1,-1),
            index=close[sampling_indices].index
        )
    def __repr__(self):
        return f"RSI(rsi_period={self.rsi_period})"
        
class Labeler(BaseEstimator,TransformerMixin):
    def __init__(self,close:pd.Series,sampling_indices:pd.Index,ptsl_scalers:tuple[int,int],min_target:float,ewma_window:int,primary_model,num_days_exit:int) -> None:
        self.close = close
        self.sampling_indices = sampling_indices
        self.ptsl_scalers = ptsl_scalers
        self.min_target = min_target
        self.ewma_window = ewma_window
        self.primary_model = primary_model
        self.num_days_exit = num_days_exit

    def fit(self,X=None,y=None):
        ## Primary Labeling

        # Triple Barrier

        vertical_barriers = self.close.index.searchsorted(self.close[self.sampling_indices].index)+self.num_days_exit
        vertical_barriers[vertical_barriers>len(self.close.index)-1] = len(self.close.index)-1
        vertical_barriers = self.close.index[vertical_barriers]
        vertical_barriers = pd.Series(vertical_barriers,index=self.close[self.sampling_indices].index)

        self.target = pd.Series(ewma_std_pct_change(self.close[self.sampling_indices].values,window=self.ewma_window),index=self.close[self.sampling_indices].index)

        vertical_barriers__target__first_touch = get_events(
            self.close,
            pd.Series(self.close[self.sampling_indices].index,index=self.close[self.sampling_indices].index),
            self.ptsl_scalers,
            self.target,
            self.min_target,
            None,
            vertical_barriers,
            )
        return__labels = get_labels(vertical_barriers__target__first_touch,self.close)
        # Primary Model
        self.side = self.primary_model.predict(self.close,self.sampling_indices)
        # print(f"Primary Model long percentage : {len(side[side==1])/len(side)}")

        ## Meta-Labeling
        vertical_barriers__target__first_touch = get_events(
            self.close,
            pd.Series(self.close[self.sampling_indices].index,index=self.close[self.sampling_indices].index),
            self.ptsl_scalers,
            self.target,
            self.min_target,
            None,
            pd.Series(vertical_barriers,index=self.close[self.sampling_indices].index),
            side=self.side
            )
        self.first_touch_time = vertical_barriers__target__first_touch['first_touch_index']
        actual_returns__meta_labels = get_labels(vertical_barriers__target__first_touch,self.close)
        self.meta_labels = actual_returns__meta_labels['meta_label']
        self.actual_returns = actual_returns__meta_labels['return']
        return self
    
def get_score(wrapper,X,sharpe,commission_pct):
    allocations = wrapper.predict_proba(X)
    actual_returns = wrapper.test_labeler.actual_returns
    if sharpe:
        return ((actual_returns*allocations).mean()-commission_pct)/(actual_returns*allocations).std()
    return (actual_returns*allocations).mean()-commission_pct

from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    def __init__(self,n_splits:int=3,first_touch_time:pd.Series=None,pct_embargo:float=0.01) -> None:
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.first_touch_time = first_touch_time
        self.pct_embargo = pct_embargo
    def split(self,X,y=None,groups=None):
        if not isinstance(self.first_touch_time,pd.Series):
            raise Exception("first_touch_time should be a pandas series")
        if (X.index == self.first_touch_time.index).all():
            pass
        else:
            raise Exception("X and first_touch_time should have the same index")
        unit = int(len(X) / self.n_splits)
        pivots = [i*unit for i in range(self.n_splits)] + [len(X)-1]
        embargo = int(len(X) * self.pct_embargo)
        for i in range(1,len(pivots)):
            first_train_indices = X.index.searchsorted(self.first_touch_time[self.first_touch_time<=X.index[pivots[i-1]]].index)
            second_train_indices = X.index.searchsorted(X.index[X.index.searchsorted(self.first_touch_time.iloc[pivots[i-1]:pivots[i]].max())+embargo:])
            train_indices = np.concatenate((first_train_indices,second_train_indices),axis=0)
            test_indices = X.index.searchsorted(X.index[pivots[i-1]:pivots[i]])
            yield train_indices,test_indices


def split_dictionary_list(arr:list[dict]):
    res = []
    def split_dictionary(new_d,d):
        if new_d.keys() == d.keys():
            return [new_d]
        res = []
        for k,v in d.items():
            if k not in new_d.keys():
                if isinstance(v,list):
                    for i in range(len(v)):
                        new_d_i = dict(new_d)
                        new_d_i[k] = v[i]
                        res.extend(split_dictionary(new_d_i,d))
                else:
                    new_d[k] = v
                    res.extend(split_dictionary(new_d,d))
                break
        return res
    for d in arr:
        res.extend(split_dictionary({},d))
    return res

import random
class CustomCV:
    def __init__(self,estimator,params_grid,cv,scoring,n_iters=0) -> None:
        self.estimator = estimator
        self.params_grid = split_dictionary_list(params_grid)
        if n_iters:
            self.params_grid = random.sample(self.params_grid,n_iters)
        self.cv = cv
        self.scoring = scoring
    def fit(self,df,y=None):
        self.estimator.fit(df)
        self.cv.first_touch_time = self.estimator.labeler.first_touch_time.copy(deep=True)
        self.cv.first_touch_time = self.cv.first_touch_time.reindex(df.index,method="bfill").fillna(df.index[-1])
        self.estimator.sampling_dates = df[self.estimator.sampling_indices].index
        self.params_scores = {}
        self.best_score_ = -100
        for params in self.params_grid:
            print(params)
            self.estimator.set_params(**params)
            score = self.scoring(df,self.estimator,self.cv)
            self.params_scores[str(params)] = score
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
            print("Score : ",score)
        # store highest (score,parameter)
        return self
    
def cross_val_score_mean_return(df,estimator,cv):
    score = 0
    for train_index,test_index in cv.split(df):
        # fit
        estimator.fit(df.iloc[train_index])
        # score
        score += (get_score(estimator,df.iloc[test_index],False,0.001))
    return score/cv.n_splits
    

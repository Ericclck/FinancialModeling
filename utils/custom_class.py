import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from utils.labeling import get_events,get_labels,ewma_std_pct_change
from utils.bars import *
from pandas import Timestamp, DateOffset
from utils.sample_weights import *
import copy
from fracdiff.sklearn import Fracdiff, FracdiffStat
MIN_NUM_SAMPLES = 100

class ClassicSampler:
    def __init__(self,cols=["open","high","low","close","volume","VWAP"],shuffle=None,verbose=0):
        self.cols = cols
        self.shuffle = shuffle
        self.verbose = verbose
    def sampling(self,df,sampling_indices):
        filtered_df = pd.DataFrame(index=df.iloc[sampling_indices].index)
        for first,last in zip(filtered_df.index[:-1],filtered_df.index[1:]):
            if ("open" in self.cols):filtered_df.loc[last,'open'] = df.loc[first,'open']
            if ("high" in self.cols):filtered_df.loc[last,'high'] = df.loc[first:last,'high'].max()
            if ("low" in self.cols):filtered_df.loc[last,'low'] = df.loc[first:last,'low'].min()
            if ("close" in self.cols):filtered_df.loc[last,'close'] = df.loc[last,'close']
            if ("volume" in self.cols):filtered_df.loc[last,'volume'] = df.loc[first:last,'volume'].sum()
            if ("VWAP" in self.cols):filtered_df.loc[last,'VWAP'] = ((df.loc[first:last,'close']*df.loc[first:last,'volume'])/df.loc[first:last,'volume'].sum()).sum()
        # since the first sampling date doesn't have a sample , it should be dropped
        filtered_df = filtered_df.iloc[1:]
        if self.verbose > 0: print("Before shuffling: ",filtered_df)
        # if shuffling is enabled
        if (self.shuffle == "open"): filtered_df["open"] = np.random.permutation(filtered_df["open"].values)
        if (self.shuffle == "high"): filtered_df["high"] = np.random.permutation(filtered_df["high"].values)
        if (self.shuffle == "low"): filtered_df["low"] = np.random.permutation(filtered_df["low"].values)
        if (self.shuffle == "close"): filtered_df["close"] = np.random.permutation(filtered_df["close"].values)
        if (self.shuffle == "volume"): filtered_df["volume"] = np.random.permutation(filtered_df["volume"].values)
        if (self.shuffle == "VWAP"): filtered_df["VWAP"] = np.random.permutation(filtered_df["VWAP"].values)
        if self.verbose > 0: print("After shuffling: ",filtered_df)

        self.sampling_cols = filtered_df.columns
        return filtered_df
    
class EventSampler(ClassicSampler):
    def __init__(self,cols=["close"],shuffle=None,verbose=0):
        self.cols = cols
        self.shuffle = shuffle
        self.verbose = verbose
    def sampling(self,df,sampling_indices):
        filtered_df = pd.DataFrame(index=df.iloc[sampling_indices].index)
        for first,last in zip(filtered_df.index[:-1],filtered_df.index[1:]):
            if ("open" in self.cols):filtered_df.loc[last,'open'] = df.loc[first,'close']
            if ("high" in self.cols):filtered_df.loc[last,'high'] = df.loc[first:last,'close'].max()
            if ("low" in self.cols):filtered_df.loc[last,'low'] = df.loc[first:last,'close'].min()
            if ("close" in self.cols):filtered_df.loc[last,'close'] = df.loc[last,'close']
            if ("balance" in self.cols):filtered_df.loc[last,'balance_mean'] = df.loc[first:last,'balance'].mean()
            if ("balance" in self.cols):filtered_df.loc[last,'balance_last'] = df.loc[first:last,'balance'].iloc[-1]
            if ("net_flow" in self.cols):filtered_df.loc[last,'net_flow'] = df.loc[first:last,'net_flow'].sum()

        # since the first sampling date doesn't have a sample , it should be dropped
        filtered_df = filtered_df.fillna(method="bfill")
        if self.verbose > 0: print("Before shuffling: ",filtered_df)
        # if shuffling is enabled
        if (self.shuffle == "open"): filtered_df["open"] = np.random.permutation(filtered_df["open"].values)
        if (self.shuffle == "high"): filtered_df["high"] = np.random.permutation(filtered_df["high"].values)
        if (self.shuffle == "low"): filtered_df["low"] = np.random.permutation(filtered_df["low"].values)
        if (self.shuffle == "close"): filtered_df["close"] = np.random.permutation(filtered_df["close"].values)
        if self.verbose > 0: print("After shuffling: ",filtered_df)

        self.sampling_cols = filtered_df.columns
        return filtered_df
    
class Labeler(BaseEstimator,TransformerMixin):
    def __init__(self,ptsl_scalers:tuple[int,int],min_target:float,ewma_window:int,primary_model,num_days_exit:int) -> None:
        self.ptsl_scalers = ptsl_scalers
        self.min_target = min_target
        self.ewma_window = ewma_window
        self.primary_model = primary_model
        self.num_days_exit = num_days_exit

    def label(self,close:pd.Series,sampling_indices:pd.Index):
        ## Primary Labeling

        # Triple Barrier

        vertical_barriers = close.index.searchsorted(close[sampling_indices].index)+self.num_days_exit
        vertical_barriers[vertical_barriers>len(close.index)-1] = len(close.index)-1
        vertical_barriers = close.index[vertical_barriers]
        vertical_barriers = pd.Series(vertical_barriers,index=close[sampling_indices].index)

        self.target = pd.Series(ewma_std_pct_change(close.values,window=self.ewma_window),index=close.index)

        vertical_barriers__target__first_touch = get_events(
            close,
            pd.Series(close[sampling_indices].index,index=close[sampling_indices].index),
            self.ptsl_scalers,
            self.target,
            self.min_target,
            None,
            vertical_barriers,
            )
        return__labels = get_labels(vertical_barriers__target__first_touch,close)
        # Primary Model
        self.side = self.primary_model.predict(close,sampling_indices)
        if len(self.side[self.side==1])/len(self.side) > 0.8 or len(self.side[self.side==1])/len(self.side) < 0.2:
            Warning(f"Primary Model long percentage : {len(self.side[self.side==1])/len(self.side)}, This may be a bad model")
        ## Meta-Labeling
        vertical_barriers__target__first_touch = get_events(
            close,
            pd.Series(close[sampling_indices].index,index=close[sampling_indices].index),
            self.ptsl_scalers,
            self.target,
            self.min_target,
            None,
            pd.Series(vertical_barriers,index=close[sampling_indices].index),
            side=self.side
            )
        self.first_touch_time = vertical_barriers__target__first_touch['first_touch_index']
        actual_returns__meta_labels = get_labels(vertical_barriers__target__first_touch,close)
        self.meta_labels = actual_returns__meta_labels['meta_label']
        self.actual_returns = actual_returns__meta_labels['return']
        return self
    
class FeatureEngineer(BaseEstimator,TransformerMixin):
    def __init__(self,sampler:EventSampler,X_pipe,shuffle_after_preprocessing_column_index:int=None,frac_stat:bool=False):
        self.sampler = sampler
        self.X_pipe = X_pipe
        self.shuffle_after_preprocessing_column_index = shuffle_after_preprocessing_column_index
        self.frac_stat = frac_stat
    def engineer(self,df,y,sampling_indices):
        sampled_df = self.sampler.sampling(df,sampling_indices).loc[y.index]
        if (self.frac_stat):
            fd = FracdiffStat()
            fd.fit(sampled_df)
            print(f"Fractional differentiation parameter : {fd.d_}")
        X_preprocessed = self.X_pipe.fit_transform(sampled_df)
        # To allow PCA MDA, which in turns is to prevent feature importance dilution
        if self.shuffle_after_preprocessing_column_index:
            X_preprocessed[:,self.shuffle_after_preprocessing_column_index] = np.random.permutation(X_preprocessed[:,self.shuffle_after_preprocessing_column_index])
        return X_preprocessed
    
class Signaler(BaseEstimator,TransformerMixin):
    def __init__(self) -> None:
        pass
    def get_event(self,close:pd.Series):
        pass

class CumulativeSumSignaler(Signaler):
    def __init__(self,cusum_threshold:float,is_pct:bool,event_col:str="close") -> None:
        self.cusum_threshold = cusum_threshold
        self.is_pct = is_pct
        self.event_col = event_col
    def get_event(self,df:pd.DataFrame) -> np.ndarray:
        event = df[self.event_col].copy(deep=True)
        if self.is_pct:
            event = event.pct_change().fillna(method="bfill")
        t_events,sp,sn = np.zeros(len(event),dtype=np.bool),0,0
        for i in range(1,len(event)):
            sp , sn = max(0,sp+event.iloc[i]-event.iloc[i-1]),min(0,sn+event.iloc[i]-event.iloc[i-1])
            if sp >= self.cusum_threshold:
                t_events[i] = True
                sp = 0
            elif sn <= -self.cusum_threshold:
                t_events[i] = True
                sn = 0
        if sum(t_events)/len(event) < 0.2 or sum(t_events)/len(event) > 0.5:
            raise Warning(f"Num of signals over entire series : {sum(t_events)/len(event)}, it should be between 0.2 and 0.5, current cusum_threshold : {self.cusum_threshold}")
        return t_events
    
class ClassifierWrapper(BaseEstimator,ClassifierMixin):
    def __init__(self,signaler:Signaler,model,enable_sample_weight:bool,labeler:Labeler,feature_engineer:FeatureEngineer,sampling_dates=None) -> None:
        self.signaler = signaler
        self.model = model
        self.sampling_dates = sampling_dates
        self.enable_sample_weight = enable_sample_weight
        self.labeler = labeler
        self.test_labeler = copy.deepcopy(labeler)
        self.feature_engineer = feature_engineer
    def fit(self,df,y=None):
        # print("Fit over date : ", df.index[0], " to ", df.index[-1])
        close = df.close.copy(deep=True)
        # Signal generation
        if self.sampling_dates is None:
            print("Warning : Only use this when first or not performing cross validation")
            self.sampling_indices = self.signaler.get_event(df)
        else:
            if not isinstance(self.sampling_dates,pd.Index) or not isinstance(self.sampling_dates,pd.DatetimeIndex):
                raise Exception("sampling_dates should be a pandas Index or a pandas DatetimeIndex")
            self.sampling_indices = close.index.searchsorted(self.sampling_dates[self.sampling_dates.isin(close.index)])
        if len(self.sampling_indices) < MIN_NUM_SAMPLES:
            raise Exception(f"Not enough samples after cusum sampling, {len(self.sampling_indices)} samples found, {MIN_NUM_SAMPLES} samples required")
        
        # labeling
        self.labeler.label(close,self.sampling_indices)
        y = self.labeler.meta_labels
        if len(y) < MIN_NUM_SAMPLES:
            raise Exception(f"Not enough samples after meta-labeling, {len(y)} samples found, {MIN_NUM_SAMPLES} samples required, Number of samples was {len(self.sampling_indices)} before meta-labeling")
        # feature engineering
        self.X_preprocessed = self.feature_engineer.engineer(df,y,self.sampling_indices)

        # modeling
        if self.enable_sample_weight: self.model.fit(self.X_preprocessed,y,sample_weight=get_sample_weights(self.labeler.first_touch_time,close))
        else: self.model.fit(self.X_preprocessed,y)
        return self
    def preprocessed_before_predict(self,df):
        # print("Predict over date : ", df.index[0], " to ", df.index[-1])
        close = df.close.copy(deep=True)

        # Signal generation
        self.test_sampling_indices = close.index.searchsorted(self.sampling_dates[self.sampling_dates.isin(close.index)])
        
        # labeling
        self.test_labeler.label(close,self.test_sampling_indices)
        y = self.test_labeler.meta_labels

        # feature engineering
        return self.feature_engineer.engineer(df,y,self.test_sampling_indices)
    def predict(self,df):
        if self.sampling_dates is None:
            print("sampling_dates not provided,since test data should be of the same sampling sequence of train data, test data is assumed to be the same as train data.")
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
        
def get_score(wrapper,X,sharpe,commission_pct):
    allocations = wrapper.predict_proba(X)
    actual_returns = wrapper.test_labeler.actual_returns
    if sharpe:
        return ((actual_returns*allocations).mean()-commission_pct)/(actual_returns*allocations).std()
    return (actual_returns*allocations).mean()-commission_pct



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
        self.estimator.sampling_dates = df.iloc[self.estimator.sampling_indices].index
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
    

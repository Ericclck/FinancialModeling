import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from utils.labeling import get_events,get_labels,ewma_std_pct_change
from utils.bars import *
from pandas import Timestamp, DateOffset
from utils.sample_weights import *
import copy
from fracdiff.sklearn import Fracdiff, FracdiffStat
import matplotlib.pyplot as plt
from utils.visualizer import Visualizer
from utils.primary_models import *
from scipy import stats
MIN_NUM_SAMPLES = 50

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
    def __init__(self,cols=["close"],lags : list[int]=None,shuffle=None,verbose=0):
        self.cols = cols
        self.lags = lags
        self.shuffle = shuffle
        self.verbose = verbose
    def sampling(self,df,sampling_indices):
        filtered_df = pd.DataFrame(index=df.iloc[sampling_indices].index)
        for first,last in zip(filtered_df.index[:-1],filtered_df.index[1:]):
            if ("open" in self.cols):filtered_df.loc[last,'open'] = df.loc[first,'close']
            if ("high" in self.cols):filtered_df.loc[last,'high'] = df.loc[first:last,'close'].max()
            if ("low" in self.cols):filtered_df.loc[last,'low'] = df.loc[first:last,'close'].min()
            if ("close" in self.cols or self.lags is not None):filtered_df.loc[last,'close'] = df.loc[last,'close']
            # for each cols in cols except close, sample last
            for col in self.cols:
                if col not in ["close","open","high","low"]:
                    filtered_df.loc[last,col] = df.loc[last,col]
        # since the first sampling date doesn't have a sample , it should be dropped
        filtered_df = filtered_df.fillna(method="bfill")
        if self.verbose > 0: print("Before shuffling: ",filtered_df)
        # if shuffling is enabled
        if (self.shuffle == "open"): filtered_df["open"] = np.random.permutation(filtered_df["open"].values)
        if (self.shuffle == "high"): filtered_df["high"] = np.random.permutation(filtered_df["high"].values)
        if (self.shuffle == "low"): filtered_df["low"] = np.random.permutation(filtered_df["low"].values)
        if (self.shuffle == "close"): filtered_df["close"] = np.random.permutation(filtered_df["close"].values)
        for col in self.cols:
            if col not in ["close","open","high","low"]:
                if (self.shuffle == col): filtered_df[col] = np.random.permutation(filtered_df[col].values)
        if self.verbose > 0: print("After shuffling: ",filtered_df)

        if self.lags:
            pct = filtered_df["close"].pct_change().fillna(0)
            for lag in self.lags:
                filtered_df[f"close_lag_{lag}"] = pct.shift(lag)
            filtered_df = filtered_df.fillna(0)
            if "close" not in self.cols: filtered_df.drop("close",axis=1,inplace=True)

        # To ensure correct order of columns
        self.sampling_cols = filtered_df.columns
        return filtered_df
    def __repr__(self) -> str:
        return f"EventSampler(cols={self.cols},lags={self.lags})"
    

#TODO: reindex by sampling_dates
class Labeler(BaseEstimator,TransformerMixin):
    def __init__(self,ptsl_scalers:tuple[int,int]=None,min_target:float=None,ewma_window:int=None,primary_model=None,num_days_exit:int=None) -> None:
        self.ptsl_scalers = ptsl_scalers
        self.min_target = min_target
        self.ewma_window = ewma_window
        self.primary_model = primary_model
        self.num_days_exit = num_days_exit
        self.target = None
        self.vertical_barriers = None
        self.asset_returns = None
        self.side = None
        self.first_touch_time = None
        self.meta_labels = None
        self.actual_returns = None

    def label(self,data:pd.DataFrame,sampling_indices:pd.Index,is_test:bool):
        close = data.close
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
        self.vertical_barriers = vertical_barriers__target__first_touch['vertical_barrier']
        return__labels = get_labels(vertical_barriers__target__first_touch,close)
        self.asset_returns_in_signal = return__labels['return']
        # Primary Model
        self.side = self.primary_model.predict(data,sampling_indices,is_test)
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
        self.close = data.close
        return self
    
class FeatureEngineer(BaseEstimator,TransformerMixin):
    def __init__(self,sampler:EventSampler=None,X_pipe=None,shuffle_after_preprocessing_column_index:int=None,frac_stat:bool=False) -> None:
        self.sampler = sampler
        self.X_pipe = X_pipe
        self.shuffle_after_preprocessing_column_index = shuffle_after_preprocessing_column_index
        self.frac_stat = frac_stat
        self.d = None
    def engineer(self,df,labeler,sampling_indices,is_test):
        y = labeler.meta_labels
        sampled_df = self.sampler.sampling(df,sampling_indices).loc[y.index]
        if (self.frac_stat):
            if is_test:
                if self.d is None: raise Exception("Fractional Differentiation not fitted")
                X_preprocessed = pd.DataFrame(index = sampled_df.index)
                # Loop over each column in the DataFrame
                for i, column in enumerate(sampled_df.columns):
                    # Initialize Fracdiff with the order for the current column
                    fd = Fracdiff(self.d[i])
                    
                    # Reshape the column to be 2D (required by sklearn-style transformers)
                    column_data = sampled_df[column].values.reshape(-1, 1)
                    
                    # Apply the Fracdiff transformer and flatten the output back to 1D
                    transformed_data = fd.fit_transform(column_data).flatten()
                    
                    # Add the transformed data to the new DataFrame
                    X_preprocessed[column] = transformed_data

                X_preprocessed = X_preprocessed.values
            else:
                fd = FracdiffStat(upper=2)
                X_preprocessed = fd.fit_transform(sampled_df)
                self.d = fd.d_
                ## There is a bug/ implicit constraint in frac_diff that d must be less than 1, had to fix in the source code if not it will throw an error
                # print("Fractional Differentiation : ",self.d)
        else:
            print("Warning : Not using fractional differentiation")
        # To allow PCA MDA, which in turns is to prevent feature importance dilution
        if self.shuffle_after_preprocessing_column_index:
            X_preprocessed[:,self.shuffle_after_preprocessing_column_index] = np.random.permutation(X_preprocessed[:,self.shuffle_after_preprocessing_column_index])
        return X_preprocessed
    
class FeatureEngineerPostLabeling(FeatureEngineer):
    def __init__(self,sampler:EventSampler=None,X_pipe=None,shuffle_after_preprocessing_column_index:int=None,frac_stat:bool=False,post_labeling_primary_model:PostLabelingPrimaryModel =  None) -> None:
        super().__init__(sampler,X_pipe,shuffle_after_preprocessing_column_index,frac_stat)
        self.post_labeling_primary_model = post_labeling_primary_model
    def engineer(self,df,labeler,sampling_indices,is_test):
        X_preprocessed = super().engineer(df,labeler,sampling_indices,is_test)
        sampled_df = pd.DataFrame(X_preprocessed,index=labeler.meta_labels.index,columns=self.sampler.sampling_cols)

        if self.post_labeling_primary_model:
            # changes content of labeler
            # TODO: wrap this inside labeler
            labeler.side = self.post_labeling_primary_model.predict(df,sampled_df,labeler,is_test)
            if len(labeler.side[labeler.side==1])/len(labeler.side) > 0.8 or len(labeler.side[labeler.side==1])/len(labeler.side) < 0.2:
                Warning(f"Primary Model long percentage : {len(labeler.side[labeler.side==1])/len(labeler.side)}, This may be a bad model")
            events = pd.concat([labeler.first_touch_time,labeler.vertical_barriers,labeler.side],axis=1)
            actual_returns__meta_labels = get_labels(events,df['close'])
            labeler.meta_labels = actual_returns__meta_labels['meta_label']
            labeler.actual_returns = actual_returns__meta_labels['return']

        return sampled_df

class FeatureEngineerWithBucketing(FeatureEngineer):
    def __init__(self, sampler: EventSampler=None, X_pipe=None, shuffle_after_preprocessing_column_index: int = None, frac_stat: bool = False,bucket_cols=None,num_buckets=0,flip=False,filter=None,bucket_return=False,verbose=0) -> None:
        super().__init__(sampler, X_pipe, shuffle_after_preprocessing_column_index, frac_stat)
        self.bucket_cols = bucket_cols
        self.num_buckets = num_buckets
        self.flip = flip
        self.filter = filter
        self.bucket_return = bucket_return
        self.feature_to_quantiles = {}
        self.buckets_to_return = {}
        self.verbose = verbose
    def engineer(self,df,labeler,sampling_indices,is_test):
        X_preprocessed = super().engineer(df,labeler,sampling_indices,is_test)
        sampled_df = pd.DataFrame(X_preprocessed,index=labeler.meta_labels.index,columns=self.sampler.sampling_cols)
        if self.bucket_cols:
            if is_test:
                #label according to features_to_bucket
                for col in self.bucket_cols:
                    sampled_df[col+"_bucket"] = pd.cut(sampled_df[col],bins=self.feature_to_quantiles[col],labels=False)
                # if self.verbose > 0:
                # print("TEST : ")
                # print(sampled_df["bucket_return"].value_counts())
            else:
                # store quantiles to features_to_bucket
                for col in self.bucket_cols:
                    quantiles = sampled_df[col].quantile(np.linspace(0,1,self.num_buckets+1)).values
                    quantiles[0] = -np.inf
                    quantiles[-1] = np.inf
                    self.feature_to_quantiles[col] = quantiles
                    sampled_df[col+"_bucket"] = pd.cut(sampled_df[col],bins=self.feature_to_quantiles[col],labels=False)
                # merge with actual returns
                sampled_df = sampled_df.merge(labeler.actual_returns,left_index=True,right_index=True)
                # store return as map
                self.buckets_to_return = sampled_df.groupby([col+"_bucket" for col in self.bucket_cols])["return"].mean().to_dict()

                sampled_df.drop("return",axis=1,inplace=True)
            try:
                # get return from table
                bucket_return = sampled_df[[col+"_bucket" for col in self.bucket_cols]].apply(lambda x: self.buckets_to_return[tuple(x)],axis=1)
            except:
                raise Exception(f"A combination of feature bucket does not exist, too granular buckets, try reducing num_buckets, current num_buckets : {self.num_buckets}, or try reducing the number of features to bucket, current features to bucket : {self.bucket_cols}")
            # remove bucket cols
            sampled_df = sampled_df.drop([col+"_bucket" for col in self.bucket_cols],axis=1)
        ## Flipping primary model
        if self.flip:
            labeler.meta_labels = labeler.meta_labels.where(bucket_return>0,1-labeler.meta_labels)
        ## Filter
        if self.filter:
            labeler.meta_labels = labeler.meta_labels.where((bucket_return<-self.filter[0]) | (bucket_return>self.filter[1]),np.nan).dropna()
            labeler.meta_labels = labeler.meta_labels.where(bucket_return>0,1-labeler.meta_labels)
            labeler.first_touch_time = labeler.first_touch_time.loc[labeler.meta_labels.index]
            labeler.actual_returns = labeler.actual_returns.loc[labeler.meta_labels.index]
            sampled_df = sampled_df.loc[labeler.meta_labels.index]
        ## Add bucket return
        if self.bucket_return:
            sampled_df["bucket_return"] = bucket_return
        if self.verbose:
            sampled_df[["balance_1k_usd"]].plot(alpha=0.5,figsize=(10,7),title="sampled_df")
            sampled_df[['net_flow_native', 'close']].plot(alpha=0.5,figsize=(10,7),title="sampled_df")
            sampled_df[["balance_1k_native"]].plot(alpha=0.5,figsize=(10,7),title="sampled_df")
            plt.show()
        return sampled_df
    
class Signaler(BaseEstimator,TransformerMixin):
    def __init__(self) -> None:
        pass
    def get_event(self,close:pd.Series):
        pass

class AllSignaler(Signaler):
    def __init__(self) -> None:
        pass
    def get_event(self,df:pd.DataFrame) -> np.ndarray:
        return np.ones(len(df),dtype=np.bool)
class CumulativeSumSignaler(Signaler):
    def __init__(self,cusum_threshold:float=None,event_col:str="close") -> None:
        self.cusum_threshold = cusum_threshold
        self.event_col = event_col
    def get_event(self,df:pd.DataFrame) -> np.ndarray:
        event = df[self.event_col].copy(deep=True)
        fd = FracdiffStat()
        event = fd.fit_transform(event.to_frame())[10:,0]
        event = pd.Series(np.concatenate([np.zeros(10),event],axis=0),index=df.index).replace(0,np.nan).fillna(method="bfill")
        # event.plot(title=f"Fractional Differentiation : {fd.d_[0]}")
        # plt.show()
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
    def __init__(self,signaler:Signaler,model,enable_sample_weight:bool,labeler:Labeler,feature_engineer:FeatureEngineer,sampling_dates=None,visualizer=None) -> None:
        self.signaler = signaler
        self.model = model
        self.sampling_dates = sampling_dates
        self.enable_sample_weight = enable_sample_weight
        self.labeler = labeler
        self.feature_engineer = feature_engineer
        self.visualizer = visualizer
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
        self.labeler.label(df,self.sampling_indices,is_test=False)
        if len(self.labeler.meta_labels) < MIN_NUM_SAMPLES:
            raise Exception(f"Not enough samples after meta-labeling, {len(self.labeler.meta_labels)} samples found, {MIN_NUM_SAMPLES} samples required, Number of samples was {len(self.sampling_indices)} before meta-labeling")
        # feature engineering
        self.X_preprocessed = self.feature_engineer.engineer(df,self.labeler,self.sampling_indices,is_test=False)

        if self.visualizer:
            # convert to df for visualization
            sampled_df = pd.DataFrame(self.X_preprocessed,index=self.labeler.meta_labels.index,columns=self.feature_engineer.sampler.sampling_cols)
            self.visualizer.visualize(sampled_df,self.labeler)
            raise ValueError("Turn Off visualization to stop this error.")

        # modeling
        if self.enable_sample_weight: self.model.fit(self.X_preprocessed,self.labeler.meta_labels,sample_weight=get_time_decay_sample_weights(self.labeler.first_touch_time,close,0))
        else: self.model.fit(self.X_preprocessed,self.labeler.meta_labels)
        return self
    def preprocessed_before_predict(self,df):
        # print("Predict over date : ", df.index[0], " to ", df.index[-1])
        close = df.close.copy(deep=True)

        # Signal generation
        self.test_sampling_indices = close.index.searchsorted(self.sampling_dates[self.sampling_dates.isin(close.index)])
        
        # labeling
        self.test_labeler = copy.deepcopy(self.labeler)
        self.test_labeler.label(df,self.test_sampling_indices,is_test=True)

        # feature engineering
        return self.feature_engineer.engineer(df,self.test_labeler,self.test_sampling_indices,is_test=True)
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
        
        
def get_score(wrapper,X,sharpe,commission_pct,verbose=False):
    allocations = wrapper.predict_proba(X)
    allocations = pd.Series(allocations,index=wrapper.test_labeler.first_touch_time.index,name="allocations")
    asset_returns = wrapper.test_labeler.close.pct_change().shift(-1).fillna(0)
    side = wrapper.test_labeler.side
    
    # sharpe
    def compute_average_allocation():
        first_touch_time = wrapper.test_labeler.first_touch_time
        # init a dict where key is first_touch_time.index and list is default value
        d = {k:[] for k in wrapper.test_labeler.close.index}
        for start,end in first_touch_time.items():
            # append allocation to list of key between start and end
            for k in d.keys():
                if k >= start and k <= end:
                    d[k].append(allocations.loc[start])
        # print(f"Allocations : {d}")
        # compute average allocation
        def mean(arr):
            if len(arr) == 0: return 0
            return sum(arr)/len(arr)
        avg_allocations = pd.Series(d).apply(mean)
        # print(f"Mean allocations : {avg_allocations}")

        def discretize(x:float):
            # round up to nearest 0.1
            return round(x*10)/10
        avg_allocations = avg_allocations.apply(discretize)
        # print(f"Discretized allocations : {avg_allocations}")
        
        return avg_allocations
    
    avg_allocations = compute_average_allocation()

    def compute_average_side():
        first_touch_time = wrapper.test_labeler.first_touch_time
        # init a dict where key is first_touch_time.index and list is default value
        d = {k:[] for k in wrapper.test_labeler.close.index}
        for start,end in first_touch_time.items():
            # append allocation to list of key between start and end
            for k in d.keys():
                if k >= start and k <= end:
                    d[k].append(side.loc[start])
                elif k > end:
                    break
        # print(f"Sides : {d}")
        # compute average side
        def indicator(arr):
            if len(arr) == 0: return 0
            if sum(arr) > 0: return 1
            elif sum(arr) < 0: return -1
            return 0

        avg_side = pd.Series(d).apply(indicator)
        # print(f"Mean sides : {avg_side}")
        return avg_side
    
    if len(side) != len(avg_allocations):
        # fill side 
        side = compute_average_side()
    return_series = asset_returns*side*avg_allocations

    if verbose:
        (side*avg_allocations).plot(title="Allocations")
        plt.show()

    def deduct_commission():
        # for every change in allocation, deduct commission
        change_in_allocation = avg_allocations.diff().fillna(0)
        return return_series - change_in_allocation.abs()*commission_pct

    return_series = deduct_commission()

    inactive_sharpe = return_series.mean() / return_series.std() * np.sqrt(365)

    return_series = return_series.replace(0,np.nan).dropna()

    annualized_sharpe = return_series.mean() / return_series.std() * np.sqrt(365)

    skewness = stats.skew(return_series)
    kurtosis = stats.kurtosis(return_series)

    adjusted_sharpe = annualized_sharpe - (skewness/6)*annualized_sharpe - (kurtosis/24)*annualized_sharpe**2

    def max_draw_down():
        cum_returns = (return_series+1).cumprod()
        return cum_returns.div(cum_returns.cummax()).sub(1).min()

    if verbose:
        avg_holding_period = (wrapper.test_labeler.vertical_barriers - wrapper.test_labeler.first_touch_time).dt.days.mean()
        print("Mean Returns Per day: ",return_series.mean())
        print("Annualized Sharpe ratio : ",annualized_sharpe)
        print("Adjusted Sharpe ratio : ",adjusted_sharpe)
        print("Annualized Sharpe ratio (including inactive period): ",inactive_sharpe)
        print("Max Draw Down : ",max_draw_down())
        print("Number of trades : ",len(allocations))
        print("Average allocation : ",avg_allocations.mean())
        print("Max allocation : ",avg_allocations.max())
        print("Average holding period per signal : ",avg_holding_period)
        get_concurrency(wrapper.test_labeler.close.index,wrapper.test_labeler.first_touch_time).plot(title="Concurrent Signals")
        plt.show()
        (return_series+1).cumprod().plot(title="Cumulative Returns")
        plt.show()

    if sharpe:
        return annualized_sharpe
    return return_series.mean()



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
        print("Warning : Only one signaling method and first touch time allowed")
        self.estimator.set_params(**self.params_grid[0])
        #TODO: Wrap this around in wrapper
        self.estimator.sampling_dates = None
        self.estimator.fit(df)
        self.cv.first_touch_time = self.estimator.labeler.first_touch_time.copy(deep=True)
        self.cv.first_touch_time = self.cv.first_touch_time.reindex(df.index,method="bfill").fillna(df.index[-1])
        self.estimator.sampling_dates = df.iloc[self.estimator.sampling_indices].index
        ###
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

def cross_val_score_sharpe(df,estimator,cv):
    score = 0
    for train_index,test_index in cv.split(df):
        # fit
        estimator.fit(df.iloc[train_index])
        # score
        score += (get_score(estimator,df.iloc[test_index],True,0.001))
    return score/cv.n_splits
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_text
from fracdiff.sklearn import FracdiffStat,Fracdiff

class crossing_ma:
    def __init__(self,fast_slow:tuple[int,int],col:str) -> None:
        self.fast_slow = fast_slow
        self.col = col
    def predict(self,df,sampling_indices,is_test):
        close = df[self.col]
        fast,slow = self.fast_slow
        # slow are filled with preceding terms mean
        slow_series = close.rolling(slow).mean()
        for i in range(len(slow_series)):
            slow_series.iloc[i] = slow_series.iloc[:i].mean()
        res = pd.Series(np.where(close.rolling(fast).mean()>slow_series,1,-1),index=close.index)
        res.iloc[:fast] = np.nan
        return res
    def __repr__(self):
        return f"crossing_ma(fast_slow={self.fast_slow},col=\"{self.col}\")"
    
class MACD:
    def __init__(self, short_period=12, long_period=26, signal_period=9, macd=None) -> None:
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.macd = macd

    def predict(self, df, sampling_indices, is_test):
        close = df.close
        
        def calculate_macd(data, short_period, long_period, signal_period):
            short_ema = data.ewm(span=short_period, adjust=False).mean()
            long_ema = data.ewm(span=long_period, adjust=False).mean()
            macd_line = short_ema - long_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            macd = macd_line - signal_line
            return macd
        
        if self.macd:
            macd = self.macd
        else:
            macd = calculate_macd(close, self.short_period, self.long_period, self.signal_period)
        return pd.Series(
            np.where(macd > 0, 1, -1),
            index=close.index
        )

    def __repr__(self):
        return f"MACD(short_period={self.short_period}, long_period={self.long_period}, signal_period={self.signal_period})"
    
class inverse_crossing_ma:
    def __init__(self,fast_slow:tuple[int,int],col:str) -> None:
        self.fast_slow = fast_slow
        self.col = col
    def predict(self,df,sampling_indices,is_test):
        close = df[self.col]
        fast,slow = self.fast_slow
        # slow are filled with preceding terms mean
        slow_series = close.rolling(slow).mean()
        for i in range(len(slow_series)):
            slow_series.iloc[i] = slow_series.iloc[:i].mean()
        res = pd.Series(np.where(close.rolling(fast).mean()<slow_series,1,-1),index=close.index)
        res.iloc[:fast] = np.nan
        return res
    def __repr__(self):
        return f"inverse_crossing_ma(fast_slow={self.fast_slow},col=\"{self.col}\")"
    
class RSI:
    def __init__(self,rsi_period,rsi=None) -> None:
        self.rsi_period = rsi_period
        self.rsi = rsi
    def predict(self,df,sampling_indices,is_test):
        close = df.close
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
            rsi = self.rsi
        else:
            rsi = calculate_rsi(close,self.rsi_period)
        return pd.Series(
            np.where(rsi<50,1,-1),
            index=close.index
        )
    def __repr__(self):
        return f"RSI(rsi_period={self.rsi_period})"
    
class BollingerBands:
    def __init__(self, ma_period=20, std_dev_multipler=2, bands=None) -> None:
        self.ma_period = ma_period
        self.std_dev_multipler = std_dev_multipler
        self.bands = bands

    def predict(self, df, sampling_indices, is_test):
        close = df.close
        
        def calculate_bands(data, ma_period, std_dev_multipler):
            ma = data.rolling(window=ma_period).mean()
            std_dev = data.rolling(window=ma_period).std()
            upper_band = ma + std_dev_multipler * std_dev
            lower_band = ma - std_dev_multipler * std_dev
            return upper_band, lower_band
        
        if self.bands:
            upper_band, lower_band = self.bands
        else:
            upper_band, lower_band = calculate_bands(close, self.ma_period, self.std_dev_multipler)
        
        return pd.Series(
            np.where(close > upper_band, 1, np.where(close < lower_band, -1, 0)),
            index=close.index
        )

    def __repr__(self):
        return f"BollingerBands(ma_period={self.ma_period}, std_dev_multipler={self.std_dev_multipler})"

class StochasticOscillator:
    def __init__(self, lookback_period=14, osc=None) -> None:
        self.lookback_period = lookback_period
        self.osc = osc

    def predict(self, df, sampling_indices, is_test):
        close = df.close

        def calculate_oscillator(data, lookback_period):
            low_min = data.rolling(window=lookback_period).min()
            high_max = data.rolling(window=lookback_period).max()

            osc = (data - low_min) / (high_max - low_min)
            return osc
        
        if self.osc:
            osc = self.osc
        else:
            osc = calculate_oscillator(close, self.lookback_period)
        
        return pd.Series(
            np.where(osc > 0.8, -1, np.where(osc < 0.2, 1, 0)),
            index=close.index
        )

    def __repr__(self):
        return f"StochasticOscillator(lookback_period={self.lookback_period})"

    

class GeneralizedVotingSystem:
    def __init__(self, models) -> None:
        self.models = models

    def predict(self, df, sampling_indices, is_test):
        predictions = []

        for model in self.models:
            pred = model.predict(df, sampling_indices, is_test)
            predictions.append(pred)
        
        # Combine the predictions and apply majority voting
        combined_preds = sum(predictions)
        majority_vote_preds = combined_preds.apply(lambda x: 1 if x > 0 else -1)

        return majority_vote_preds

    def __repr__(self):
        return f"GeneralizedVotingSystem(models={self.models})"
    
class PreCalculatedSideModel:
    # DATA LEAKAGE !!! prototype only
    def __init__(self) -> None:
        pass
    def predict(self,df,sampling_dates,is_test):
        return df["pre_calculated_side"]
    def __repr__(self):
        return f"PreCalculatedSideModel()"

class PostLabelingPrimaryModel:
    def __init__(self) -> None:
        pass
    def predict(self,df,labeler,is_test):
        pass

class TreeModel(PostLabelingPrimaryModel):
    def __init__(self,model:DecisionTreeClassifier,cols:list,filtered:int) -> None:
        self.model = model
        self.cols = cols
        self.filtered = filtered
        self.d = None
    def predict(self,df,processed_df,labeler,is_test):
        if self.filtered == 2:
            # directly use processed_df
            if is_test:
                return pd.Series(self.model.predict(processed_df),index=processed_df.index,name="side")
            else:
                self.model.fit(processed_df,np.sign(labeler.asset_returns_in_signal),sample_weight=labeler.asset_returns_in_signal.abs())
                return pd.Series(self.model.predict(processed_df),index=processed_df.index,name="side")
        elif self.filtered == 1:
            sampling_dates = processed_df.index
        else:
            sampling_dates = df.index
        if is_test:
            # raise Error if model is not fitted
            if not hasattr(self.model, "tree_") or self.d is None:
                raise Exception("Model is not fitted yet")

            fd_df = pd.DataFrame(index = df.loc[sampling_dates].index)
            # Loop over each column in the DataFrame
            for i, column in enumerate(self.cols):
                # Initialize Fracdiff with the order for the current column
                fd = Fracdiff(self.d[i])
                
                # Reshape the column to be 2D (required by sklearn-style transformers)
                column_data = df.loc[sampling_dates,column].values.reshape(-1, 1)
                
                # Apply the Fracdiff transformer and flatten the output back to 1D
                transformed_data = fd.fit_transform(column_data).flatten()
                
                # Add the transformed data to the new DataFrame
                fd_df[column] = transformed_data

            ## first 10 of fd_df has to be predicted regardless
            return pd.Series(self.model.predict(fd_df[self.cols]),index=sampling_dates,name="side")
        else:
            pct = df.loc[sampling_dates].close.pct_change().shift(-1).fillna(0)
            side = np.sign(pct)

            fd = FracdiffStat(upper=2)
            fd_df = pd.DataFrame(fd.fit_transform(df.loc[sampling_dates,self.cols]),index=sampling_dates,columns=self.cols)
            self.d = fd.d_


            self.model.fit(fd_df.iloc[10:],side.iloc[10:],sample_weight=pct.iloc[10:].abs())

            # print tree using export_text
            # print(export_text(self.model,feature_names=self.cols))

            ## first 10 of fd_df has to be predicted regardless
            return pd.Series(self.model.predict(fd_df[self.cols]),index=sampling_dates,name="side")
    def __repr__(self):
        return f"TreeModel(model={self.model},cols={self.cols},filtered={self.filtered})"
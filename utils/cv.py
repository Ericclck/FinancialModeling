from sklearn.model_selection._split import _BaseKFold
import pandas as pd
import numpy as np
import itertools

class PurgedKFold(_BaseKFold):
    def __init__(self,n_splits:int=3,first_touch_time:pd.Series=None,pct_embargo:float=0.01) -> None:
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.first_touch_time = first_touch_time
        self.pct_embargo = pct_embargo
    def split(self,X,y=None,groups=None):
        if not isinstance(self.first_touch_time,pd.Series):
            raise Exception("first_touch_time should be a pandas series")
        if (X.index != self.first_touch_time.index).all():
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

from scipy.special import comb
class CombinatorialPurgedKFold(PurgedKFold):
    def __init__(self,n_splits:int=5,first_touch_time:pd.Series=None,pct_embargo:float=0.01,num_test_blocks:int=2) -> None:
        super().__init__(n_splits,first_touch_time,pct_embargo)
        if num_test_blocks > int(n_splits/2)+1:
            raise Exception("num_test_blocks should be less than or equal to int(n_splits/2)+1")
        self.num_test_blocks = num_test_blocks
        self.num_combinations = int(comb(n_splits,num_test_blocks))
    def split(self,X,y=None,groups=None):
        if not isinstance(self.first_touch_time,pd.Series):
            raise Exception("first_touch_time should be a pandas series")
        if (X.index != self.first_touch_time.index).all():
            raise Exception("X and first_touch_time should have the same index")
        tuples = [(train_indices,test_indices) for train_indices,test_indices in super().split(X,y,groups)]
        tuple_combinations = list(itertools.combinations(tuples,self.num_test_blocks))
        for i,combination in enumerate(tuple_combinations):
            train_indices = combination[0][0]
            test_indices = combination[0][1]
            for j,t in enumerate(combination[1:]):
                # train_indices should be intersections
                train_indices = np.intersect1d(train_indices,t[0])
                # test_indices should be unions
                test_indices = np.union1d(test_indices,t[1])
            yield train_indices,test_indices
                
        

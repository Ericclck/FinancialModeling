from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from fracdiff.sklearn import Fracdiff
from utils.custom_class import *
from utils.get_data import *
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot import plot_kde
from utils.analysis import perform_dw_test
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap
from utils.labeling import *
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from utils.feature_importance import *
from utils.cv import *
from utils.sequential_bootstrapping import *
import xgboost

params_grid = [
    # {
    #     "signaler__cusum_threshold": [0.4,0.6],
    #     "model":[BaggingClassifier(estimator=RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.05,max_features=int(1)),max_samples=0.8,n_jobs=-1)],
    #     "model__n_estimators" : [1000,2000,5000],
    #     "model__estimator__max_depth" : [3,10],
    #     "model__estimator__class_weight": ["balanced_subsample","balanced"],
    #     "feature_engineer__X_pipe__ImproveSkewness": [Fracdiff(0.7)],
    #     "feature_engineer__X_pipe__DimensionReduction" : [None,PCA(2)],
    #     "labeler__primary_model" : [crossing_ma((10,20)),crossing_ma((20,40)),crossing_ma((50,100))],
    #     "labeler__min_target" : [0.0005],
    #     "labeler__ptsl_scalers" : [(3,1),(2,2),(1,3),(1,5),(1,7)],
    #     "labeler__ewma_window" : [10,20,50],
    #     "enable_sample_weight" : [True,False],
    # },
    # {
    #     "signaler__cusum_threshold": [0.4,0.6],
    #     "model":[BaggingClassifier(estimator=RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.05,max_features=int(1)),max_samples=0.8,n_jobs=-1)],
    #     "model__n_estimators" : [1000,2000,5000],
    #     "model__estimator__max_depth" : [3,10],
    #     "model__estimator__class_weight": ["balanced_subsample","balanced"],
    #     "feature_engineer__X_pipe__ImproveSkewness": [Fracdiff(0.7)],
    #     "feature_engineer__X_pipe__DimensionReduction" : [None,PCA(2)],
    #     "labeler__primary_model" : [RSI(7),RSI(14),RSI(30)],
    #     "labeler__min_target" : [0.0005],
    #     "labeler__ptsl_scalers" : [(3,1),(1,3),(2,2),(1,5),(1,7)],
    #     "labeler__ewma_window" : [10,20,50],
    #     "enable_sample_weight" : [True,False],
    # },
    {
        "signaler__cusum_threshold": [0.01,0.02],
        "model":[xgboost.XGBClassifier(min_child_weight=0.05,colsample_bytree=0.1,subsample=0.75)],
        "model__learning_rate" : [0.1,0.2,0.3],
        "model__n_estimators" : [100,1000,5000],
        "model__max_depth" : [3,10],
        "feature_engineer__X_pipe__ImproveSkewness": [Fracdiff(0.7)],
        "feature_engineer__X_pipe__DimensionReduction" : [None,PCA(2)],
        "labeler__primary_model" : [crossing_ma((10,20)),crossing_ma((20,40)),crossing_ma((50,100))],
        "labeler__min_target" : [0.0005],
        "labeler__ptsl_scalers" : [(3,1),(2,2),(1,3),(1,5),(1,7)],
        "labeler__num_days_exit" : [10,20,50],
        "labeler__ewma_window" : [10,20,50],
        "enable_sample_weight" : [True,False],
    },
    {
        "signaler__cusum_threshold": [0.01,0.02],
        "model":[xgboost.XGBClassifier(min_child_weight=0.05,colsample_bytree=0.1,subsample=0.75)],
        "model__learning_rate" : [0.1,0.2,0.3],
        "model__n_estimators" : [100,1000,5000],
        "model__max_depth" : [3,10],
        "feature_engineer__X_pipe__ImproveSkewness": [Fracdiff(0.7)],
        "feature_engineer__X_pipe__DimensionReduction" : [None,PCA(2)],
        "labeler__primary_model" : [RSI(7),RSI(14),RSI(30)],
        "labeler__min_target" : [0.0005],
        "labeler__ptsl_scalers" : [(3,1),(1,3),(2,2),(1,5),(1,7)],
        "labeler__ewma_window" : [10,20,50],
        "labeler__num_days_exit" : [10,20,50],
        "enable_sample_weight" : [True,False],
    },

]

df = get_data("TLT")

X_pipe = Pipeline([
    ('ImproveSkewness', Fracdiff(0.7)),
    ('OutlierSmoother',None),
    ('Scaler',StandardScaler()),
    ('DimensionReduction',None),
])


wrapper = ClassifierWrapper(signaler=CumulativeSumSignaler(cusum_threshold=0.6,is_pct=True),
                            model=RandomForestClassifier(),
                            enable_sample_weight=False,
                            labeler=Labeler(
                                ptsl_scalers=(2,2),
                                min_target=0.001,
                                ewma_window=10,
                                primary_model=crossing_ma((10,20)),
                                num_days_exit=10
                            ),
                            feature_engineer=FeatureEngineer(ClassicSampler(cols=["open","close"]),
                                                             X_pipe=X_pipe
                                                             )
                            )
wrapper.set_params(**{'signaler__cusum_threshold': 0.01, 'model': xgboost.XGBClassifier(min_child_weight=0.05,colsample_bytree=0.1,subsample=0.75), 'model__n_estimators': 1000, 'model__max_depth': 10,"model__learning_rate":0.3,'feature_engineer__X_pipe__ImproveSkewness': Fracdiff(d=0.7, window=10), 'feature_engineer__X_pipe__DimensionReduction': None, 'labeler__primary_model': crossing_ma(fast_slow=(10, 20)), 'labeler__min_target': 0.00001, 'labeler__ptsl_scalers': (3, 1), 'labeler__ewma_window': 50, 'enable_sample_weight': False})

wrapper.fit(df.iloc[:int(len(df)*0.8)])
# MDI
# MDI(wrapper,df[:int(len(df)*0.8)],sampling_cols=["open","high","low","close","volume"])

# MDA
# MDA_matrix = MDA(wrapper,df[:int(len(df)*0.8)],cv=CombinatorialPurgedKFold(5,None,pct_embargo=0.05),sampling_cols=wrapper.feature_engineer.sampler.sampling_cols)
# print(MDA_matrix.mean(axis=0))

# from sklearn.preprocessing import StandardScaler
# wrapper.fit(df)
# wrapper.sampling_dates = df.iloc[wrapper.sampling_indices].index
# wrapper.fit(df.iloc[:int(len(df)*0.8)])

# uniqueness
# print(get_average_uniqueness(construct_indicator_matrix(df.index,wrapper.labeler.first_touch_time)).mean())

# testing data !!!
# print(get_score(wrapper,df.iloc[int(len(df)*0.8)+1:],sharpe=False,commission_pct=0.001))

# print(wrapper.predict(df))
# print(get_score(wrapper,df,sharpe=False,commission_pct=0.001))

# params_grid = [
# {'model': GradientBoostingClassifier(max_features=1, min_weight_fraction_leaf=0.05,
#                            n_estimators=2000, subsample=0.8), 'model__n_estimators': 2000, 'model__max_depth': 10, 'model__max_features': 1, 'model__min_weight_fraction_leaf': 0.05, 'X_pipe__ImproveSkewness': Fracdiff(d=0.7, window=10, mode=same, window_policy=fixed), 'X_pipe__DimensionReduction': PCA(n_components=2), 'cusum_threshold': 0.4, 'primary_model': RSI(rsi_period=7), 'min_target': 0.005, 'ptsl_scalers': (1, 3), 'ewma_window': 50, 'sample_weight': False, 'num_days_exit': 10},
# {'model': GradientBoostingClassifier(max_depth=10, max_features=1,
#                            min_weight_fraction_leaf=0.05, n_estimators=1000,
#                            subsample=0.8), 'model__n_estimators': 2000, 'model__max_depth': 3, 'model__max_features': 1, 'model__min_weight_fraction_leaf': 0.05, 'X_pipe__ImproveSkewness': Fracdiff(d=0.7, window=10, mode=same, window_policy=fixed), 'X_pipe__DimensionReduction': None, 'cusum_threshold': 0.4, 'primary_model': crossing_ma(fast_slow=(20, 40)), 'min_target': 0.005, 'ptsl_scalers': (1, 3), 'ewma_window': 50, 'sample_weight': False, 'num_days_exit': 10}
# Score :  0.0018479610105574154
# {'model': GradientBoostingClassifier(max_depth=10, max_features=1,
#                            min_weight_fraction_leaf=0.05, n_estimators=2000,
#                            subsample=0.8), 'model__n_estimators': 5000, 'model__max_depth': 10, 'model__max_features': 1, 'model__min_weight_fraction_leaf': 0.05, 'X_pipe__ImproveSkewness': Fracdiff(d=0.7, window=10, mode=same, window_policy=fixed), 'X_pipe__DimensionReduction': None, 'cusum_threshold': 0.4, 'primary_model': crossing_ma(fast_slow=(20, 40)), 'min_target': 0.005, 'ptsl_scalers': (1, 3), 'ewma_window': 20, 'sample_weight': False, 'num_days_exit': 50}
# Score :  0.0020051566640856993
# ]

ccv = CustomCV(wrapper,params_grid,cv=CombinatorialPurgedKFold(5,None,pct_embargo=0.05),scoring=cross_val_score_mean_return,n_iters=20)
# 0.8 is the train size
ccv.fit(df.iloc[:int(len(df)*0.8)])
print(ccv.best_params_)
print(ccv.best_score_)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from fracdiff.sklearn import Fracdiff, FracdiffStat
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
    {
        "signaler__cusum_threshold": [5],
        "model":[xgboost.XGBClassifier(min_child_weight=0.05,colsample_bytree=0.1,subsample=0.7)],
        "model__learning_rate" : [0.1,0.2,0.3],
        "model__n_estimators" : [100,1000,5000],
        "model__max_depth" : [3,10],
        "feature_engineer__X_pipe__ImproveSkewness": [Fracdiff(0.75),PowerTransformer()],
        "feature_engineer__X_pipe__DimensionReduction" : [None],
        "labeler__primary_model" : [crossing_ma((10,20)),crossing_ma((20,40)),crossing_ma((50,100))],
        "labeler__min_target" : [0.002],
        "labeler__ptsl_scalers" : [(3,1),(2,2),(1,3),(1,5),(1,7)],
        "labeler__ewma_window" : [10,20,50],
        "enable_sample_weight" : [True,False],
    },
    {
        "signaler__cusum_threshold": [5],
        "model":[xgboost.XGBClassifier(min_child_weight=0.05,colsample_bytree=0.1,subsample=0.7)],
        "model__learning_rate" : [0.1,0.2,0.3],
        "model__n_estimators" : [100,1000,5000],
        "model__max_depth" : [3,10],
        "feature_engineer__X_pipe__ImproveSkewness": [Fracdiff(0.75),PowerTransformer()],
        "feature_engineer__X_pipe__DimensionReduction" : [None],
        "labeler__primary_model" : [RSI(7),RSI(14),RSI(30)],
        "labeler__min_target" : [0.002],
        "labeler__ptsl_scalers" : [(3,1),(1,3),(2,2),(1,5),(1,7)],
        "labeler__ewma_window" : [10,20,50],
        "enable_sample_weight" : [True,False],
    },

]

df = pd.read_csv("data/AdrBalCnt.csv",index_col="Time",parse_dates=True)[["BTC / Price, USD","BTC / Addresses, with balance, greater than $1K, count"]]
df = df.merge(pd.read_csv("data/NetFlowBinance.csv",index_col="Time",parse_dates=True)[["BTC / Flow, net, Binance, native units"]],left_index=True,right_index=True,how="left")
df = df[df.index>"2018-01-01"].replace(0,np.nan).dropna()
df.columns = ["close","balance","net_flow"]
print("Raw df:",df)

X_pipe = Pipeline([
    ('ImproveSkewness', Fracdiff(0.7)),
    ('OutlierSmoother',None),
    ('Scaler',StandardScaler()),
    ('DimensionReduction',None),
])


wrapper = ClassifierWrapper(signaler=CumulativeSumSignaler(0.02,True,event_col="net_flow"),
                            model=RandomForestClassifier(),
                            enable_sample_weight=False,
                            labeler=Labeler(
                                ptsl_scalers=(2,2),
                                min_target=0.001,
                                ewma_window=10,
                                primary_model=crossing_ma((10,20)),
                                num_days_exit=10
                            ),
                            feature_engineer=FeatureEngineer(EventSampler(cols=["open"]),X_pipe=X_pipe)
                            )
wrapper.set_params(**{'signaler__cusum_threshold': 5,
'model': xgboost.XGBClassifier(min_child_weight=0.05,colsample_bytree=0.1,subsample=0.7),
'feature_engineer__X_pipe__DimensionReduction': None,
'labeler__primary_model': crossing_ma(fast_slow=(50, 100)), 'labeler__min_target': 0.002,
'labeler__ptsl_scalers': (1, 5),
'labeler__ewma_window': 10,
'enable_sample_weight': False})

# FracdiffStat
# wrapper.feature_engineer.frac_stat = True
# wrapper.fit(df[:int(len(df)*0.8)])

# uniqueness
# wrapper.fit(df[:int(len(df)*0.8)])
# print("Uniqueness : ",get_average_uniqueness(construct_indicator_matrix(df.index,wrapper.labeler.first_touch_time)).mean())

# MDI
# wrapper.fit(df[:int(len(df)*0.8)])
# MDI(wrapper,df[:int(len(df)*0.8)],sampling_cols=wrapper.feature_engineer.sampler.sampling_cols)

# MDA
# wrapper.fit(df[:int(len(df)*0.8)])
# MDA_matrix = MDA(wrapper,df[:int(len(df)*0.8)],cv=CombinatorialPurgedKFold(5,None,pct_embargo=0.05),sampling_cols=wrapper.feature_engineer.sampler.sampling_cols)
# print(MDA_matrix.mean(axis=0))

# testing data!!!
wrapper.fit(df)
wrapper.sampling_dates = df.iloc[wrapper.sampling_indices].index
wrapper.fit(df[:int(len(df)*0.8)])
print(get_score(wrapper,df[int(len(df)*0.8):],sharpe=False,commission_pct=0.001))


# to ensure everything make sense for entire df
# wrapper.fit(df)
# print("-----------------------------------------------")
# print("Every hyperparameter listed before make sense.")
# print("-----------------------------------------------")
# # refitted to cv
# ccv = CustomCV(wrapper,params_grid,cv=CombinatorialPurgedKFold(5,None,pct_embargo=0.05),scoring=cross_val_score_mean_return,n_iters=40)
# # 0.8 is the train size
# ccv.fit(df.iloc[:int(len(df)*0.8)])
# print(ccv.best_params_)
# print(ccv.best_score_)
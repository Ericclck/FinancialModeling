from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer,MinMaxScaler
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
from utils.visualizer import *
from utils.custom_model import *
from utils.save_tree import *
from utils.frac_diff_tools import *
from utils.OU import *


df = pd.read_csv("data/AdrBalCnt.csv",index_col="Time",parse_dates=True)
df = df.merge(pd.read_csv("data/NetFlowBinance.csv",index_col="Time",parse_dates=True),left_index=True,right_index=True)
df = df.merge(pd.read_csv("data/^VIX.csv",index_col="date",parse_dates=True)['close'],left_index=True,right_index=True,how="left")
df = df.merge(pd.read_csv("data/SPY.csv",index_col="date",parse_dates=True)['close'],left_index=True,right_index=True,how="left")
df = df.fillna(method="ffill")

df = df.drop(["BTC / Addresses, with balance, greater than 1M native units, count","BTC / Price, USD_y"],axis=1)

df = df[df.index>"2018-01-01"].replace(0,np.nan).dropna(axis=1)

df.columns = ["close",
              "balance_10k_native",
              "balance_100k_native",
              "balance_1k_native",
              "balance_1k_usd",
              "balance_10k_usd",
              "balance_100k_usd",
              "balance_1m_usd",
              "balance_10m_usd",
              "net_flow_native",
              "net_flow_usd",
              'vix_close',
              'spy_close']
print("Raw df:",df)

# frac diff df
train_test_boundary = 0.8
_,d = get_frac_diff_df(df.iloc[:int(len(df)*train_test_boundary)],df.columns)
fd_df,_ = get_frac_diff_df(df,df.columns,d=d)
fd_df['pct'] = df.close.pct_change().shift(-1)
fd_df['isLong'] = np.where(fd_df['pct']>0,1,-1)
tree = DecisionTreeClassifier(min_weight_fraction_leaf=0.2,class_weight='balanced')
tree.fit(fd_df.drop(['pct','isLong'],axis=1).iloc[:int(len(fd_df)*train_test_boundary)],fd_df['isLong'].iloc[:int(len(fd_df)*train_test_boundary)],sample_weight=fd_df['pct'].iloc[:int(len(fd_df)*train_test_boundary)].abs())
save_one_tree(tree,"plots/BTC/",fd_df.drop(['pct','isLong'],axis=1).columns)
print("Frac diff df:",fd_df)
pct = fd_df['pct']
isLong = fd_df['isLong']
fd_df = fd_df.drop(['pct','isLong'],axis=1)
side = tree.predict(fd_df)
df['pre_calculated_side'] = side
print("Side:",df.pre_calculated_side.value_counts())
# sharpe
EDA_return_series = (1+side*pct).iloc[:int(len(fd_df)*train_test_boundary)]
print("Annualized Sharpe :",(EDA_return_series.mean()-0.001)/EDA_return_series.std()*np.sqrt(365))
# log scale return 
# (1+side*pct).iloc[:int(len(fd_df)*train_test_boundary)].cumprod().plot(title="Training")
# plt.show()
# (1+side*pct).iloc[int(len(fd_df)*train_test_boundary):].cumprod().plot(title="Testing")
# plt.show()


# from scipy.optimize import minimize

# Run the optimizer
# bounds = [(0.01, 20), (-20, 20), (0.01, 20)]
# result = minimize(OU_likelihood, (5,0,1), args=(df.close.values,), bounds=bounds)

# # Extract the fitted parameters
# theta_hat, mu_hat, sigma_hat = result.x
# print(f"theta_hat = {theta_hat:.3f}, mu_hat = {mu_hat:.3f}, sigma_hat = {sigma_hat:.3f}")

params_grid = [
    # {
    #     "signaler" : [CumulativeSumSignaler(5000,event_col="balance_100k_usd")],
    #     "model":[RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.1,max_features=2,max_samples=0.554,class_weight='balanced_subsample',n_jobs=-1)
    #              ],
    #     # "model__learning_rate" : [0.1,0.2,0.3],
    #     "model__n_estimators" : [1000],
    #     "feature_engineer__X_pipe__DimensionReduction" : [None],
    #     "labeler__primary_model" : [
    #         # crossing_ma((10,20),"balance_1k_native"),
    #         # crossing_ma((5,10),"balance_1k_native"),
    #         PreCalculatedSideModel(),
    #         ],
    #     "labeler__min_target" : [0.006],
    #     "labeler__ptsl_scalers" : [(1,5),(1,7),(2,10)],
    #     "labeler__ewma_window" : [10],
    #     "labeler__num_days_exit" : [50],
    #     "enable_sample_weight" : [True],
    #     "feature_engineer__sampler" : [
    #         # EventSampler(cols=["net_flow_native","balance_1k_usd","close","balance_1k_native"]),
    #                                    EventSampler(cols=['net_flow_native', 'balance_100k_usd', 'balance_100k_native', 'balance_1k_native','balance_10k_native'],)
    #                                    ],
    #     "feature_engineer__post_labeling_primary_model":[TreeModel(DecisionTreeClassifier(min_weight_fraction_leaf=0.2,class_weight="balanced"),cols=[c for c in df.columns if c not in ["pre_calculated_side"]],filtered=True)]
    # },
    {
        "signaler" : [CumulativeSumSignaler(5000,event_col="net_flow_native")],
        "model":[RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.1,max_features=2,max_samples=0.711,class_weight='balanced_subsample',n_jobs=-1)
                 ],
        "model__n_estimators" : [1000],
        "feature_engineer__X_pipe__DimensionReduction" : [None],
        "labeler__primary_model" : [
            PreCalculatedSideModel(),
            ],
        "labeler__min_target" : [0.006],
        "labeler__ptsl_scalers" : [(1,5),(1,7),(2,10)],
        "labeler__ewma_window" : [10],
        "labeler__num_days_exit" : [50],
        "enable_sample_weight" : [True],
        "feature_engineer__sampler" : [EventSampler(cols=['balance_1k_native',   'vix_close', 'spy_close'])],
        "feature_engineer__post_labeling_primary_model":[TreeModel(DecisionTreeClassifier(min_weight_fraction_leaf=0.2,class_weight="balanced"),cols=[c for c in df.columns if c not in ["pre_calculated_side"]],filtered=1),TreeModel(model=DecisionTreeClassifier(class_weight='balanced', min_weight_fraction_leaf=0.2),cols=['close', 'balance_10k_native', 'balance_100k_native', 'balance_1k_native', 'balance_1k_usd', 'balance_10k_usd', 'balance_100k_usd', 'balance_1m_usd', 'balance_10m_usd', 'net_flow_native', 'net_flow_usd', 'vix_close', 'spy_close'],filtered=0)]
    },
    # {
    #     "signaler" : [CumulativeSumSignaler(100000,event_col="balance_1k_usd")],
    #     "model":[RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.1,max_features=3,max_samples=0.653,class_weight='balanced_subsample',n_jobs=-1)
    #              ],
    #     "model__n_estimators" : [1000],
    #     "feature_engineer__X_pipe__DimensionReduction" : [None],
    #     "labeler__primary_model" : [
    #         # crossing_ma((10,20),"balance_1k_usd"),
    #         # crossing_ma((5,10),"balance_1k_usd")
    #         PreCalculatedSideModel(),
    #         ],
    #     "labeler__min_target" : [0.006],
    #     "labeler__ptsl_scalers" : [(1,5),(1,7),(2,10)],
    #     "labeler__ewma_window" : [10],
    #     "labeler__num_days_exit" : [50],
    #     "enable_sample_weight" : [True],
    #     "feature_engineer__sampler" : [EventSampler(cols=['net_flow_native', 'balance_100k_usd', 'balance_100k_native', 'balance_1k_native','balance_10k_native',"balance_1m_usd","vix_close"])],
    # "feature_engineer__post_labeling_primary_model":[TreeModel(DecisionTreeClassifier(min_weight_fraction_leaf=0.2,class_weight="balanced"),cols=[c for c in df.columns if c not in ["pre_calculated_side"]],filtered=True)]
    # },
    # {
    #     "signaler" : [CumulativeSumSignaler(100,event_col="balance_10m_usd")],
    #     "model":[RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.1,max_features=2,max_samples=0.54,class_weight='balanced_subsample',n_jobs=-1)
    #              ],
    #     "model__n_estimators" : [1000],
    #     "feature_engineer__X_pipe__DimensionReduction" : [None],
    #     "labeler__primary_model" : [
    #         # crossing_ma((10,20),"balance_1k_usd"),
    #         # crossing_ma((5,10),"balance_1k_usd")
    #         PreCalculatedSideModel(),
    #         ],
    #     "labeler__min_target" : [0.006],
    #     "labeler__ptsl_scalers" : [(1,3),(1,7)],
    #     "labeler__ewma_window" : [10],
    #     "labeler__num_days_exit" : [50],
    #     "enable_sample_weight" : [True],
    #     "feature_engineer__sampler" : [EventSampler(cols=["net_flow_native","balance_1k_usd","close","balance_1k_native"])],
    # "feature_engineer__post_labeling_primary_model":[TreeModel(DecisionTreeClassifier(min_weight_fraction_leaf=0.2,class_weight="balanced"),cols=[c for c in df.columns if c not in ["pre_calculated_side"]],filtered=True)]
    # },
    # {
    #     "signaler" : [AllSignaler()],
    #     "model":[RandomForestClassifier(n_estimators=1,min_weight_fraction_leaf=0.1,max_features=2,max_samples=0.353,class_weight='balanced_subsample',n_jobs=-1)
    #              ],
    #     "model__n_estimators" : [1000],
    #     "feature_engineer__X_pipe__DimensionReduction" : [None],
    #     "labeler__primary_model" : [
    #         # crossing_ma((10,20),"balance_1k_usd"),
    #         # crossing_ma((5,10),"balance_1k_usd")
    #         PreCalculatedSideModel(),
    #         ],
    #     "labeler__min_target" : [0.006],
    #     "labeler__ptsl_scalers" : [(1,3),(1,7)],
    #     "labeler__ewma_window" : [10],
    #     "labeler__num_days_exit" : [50],
    #     "enable_sample_weight" : [True], 
    #     "feature_engineer__sampler" : [EventSampler(cols=["balance_10k_native","balance_1k_native","spy_close","vix_close"])],
    # "feature_engineer__post_labeling_primary_model":[TreeModel(DecisionTreeClassifier(min_weight_fraction_leaf=0.2,class_weight="balanced"),cols=[c for c in df.columns if c not in ["pre_calculated_side"]],filtered=True)]
    # },
]


X_pipe = Pipeline([
    ('ImproveSkewness', None),
    ('OutlierSmoother',None),
    ('DimensionReduction',None),
])

wrapper = ClassifierWrapper(signaler=None,
                            model=RandomForestClassifier(),
                            enable_sample_weight=None,
                            labeler=Labeler(),
                            feature_engineer=FeatureEngineerPostLabeling(X_pipe=X_pipe,frac_stat=True),
                            # visualizer=TwoDimBucketVisualizer(bucket_col=["net_flow_native","balance_1k_native"],num_buckets=3)
                            )

wrapper.set_params(**{'signaler': CumulativeSumSignaler(cusum_threshold=5000, event_col='net_flow_native'), 'model': RandomForestClassifier(class_weight='balanced_subsample', max_features=2,
                       max_samples=0.711, min_weight_fraction_leaf=0.1,
                       n_estimators=1000, n_jobs=-1), 'model__n_estimators': 1000, 'feature_engineer__X_pipe__DimensionReduction': None, 'labeler__primary_model': PreCalculatedSideModel(), 'labeler__min_target': 0.006, 'labeler__ptsl_scalers': (1, 5), 'labeler__ewma_window': 10, 'labeler__num_days_exit': 50, 'enable_sample_weight': True, 'feature_engineer__sampler': EventSampler(cols=['balance_1k_native', 'vix_close', 'spy_close'],lags=None), 'feature_engineer__post_labeling_primary_model': TreeModel(model=DecisionTreeClassifier(class_weight='balanced', min_weight_fraction_leaf=0.2),cols=['close', 'balance_10k_native', 'balance_100k_native', 'balance_1k_native', 'balance_1k_usd', 'balance_10k_usd', 'balance_100k_usd', 'balance_1m_usd', 'balance_10m_usd', 'net_flow_native', 'net_flow_usd', 'vix_close', 'spy_close'],filtered=1)})

# FracdiffStat
# wrapper.feature_engineer.frac_stat = True
# wrapper.fit(df[:int(len(df)*0.8)])

# uniqueness
# wrapper.fit(df[:int(len(df)*0.8)])
# print("Uniqueness : ",get_average_uniqueness(construct_indicator_matrix(df.index,wrapper.labeler.first_touch_time)).mean())

# MDI
wrapper.fit(df[:int(len(df)*0.8)])
MDI(wrapper,df[:int(len(df)*0.8)],sampling_cols=wrapper.feature_engineer.sampler.sampling_cols)

# # MDA
# wrapper.fit(df.iloc[:int(len(df)*0.8)])
# MDA_matrix = MDA(wrapper,df[:int(len(df)*0.8)],cv=CombinatorialPurgedKFold(5,None,pct_embargo=0.05),sampling_cols=wrapper.feature_engineer.sampler.sampling_cols)
# print(MDA_matrix.mean(axis=0))

# testing data!!!
# print("TESTING")
wrapper.fit(df)
wrapper.sampling_dates = df.iloc[wrapper.sampling_indices].index
wrapper.fit(df[:int(len(df)*0.8)])
print(get_score(wrapper,df[int(len(df)*0.8):],sharpe=True,commission_pct=0.001,verbose=1))

folder_path = 'plots/BTC/net_flow_native_tree/'
save_one_tree(wrapper.feature_engineer.post_labeling_primary_model.model,folder_path,wrapper.feature_engineer.post_labeling_primary_model.cols)
save_trees(wrapper,folder_path)

# to ensure everything make sense for entire df
wrapper.fit(df)
# print("-----------------------------------------------")
# print("Every hyperparameter listed before make sense.")
# print("-----------------------------------------------")

def signal_cv(i):
    print(f"Using signaler : {params_grid[i]['signaler']}")
    # refitted to cv
    ccv = CustomCV(wrapper,[params_grid[i]],cv=CombinatorialPurgedKFold(5,None,pct_embargo=0.05),scoring=cross_val_score_sharpe,n_iters=0)
    # 0.8 is the train size
    ccv.fit(df.iloc[:int(len(df)*0.8)])
    print(ccv.best_params_)
    print(ccv.best_score_)

for i in range(len(params_grid)):
    signal_cv(i)

# {'signaler': CumulativeSumSignaler(cusum_threshold=5000, event_col='net_flow_native'), 'model': RandomForestClassifier(class_weight='balanced_subsample', max_features=2,
#                        max_samples=0.56, min_weight_fraction_leaf=0.1,
#                        n_estimators=1000, n_jobs=-1), 'model__n_estimators': 1000, 'feature_engineer__X_pipe__DimensionReduction': None, 'labeler__primary_model': PreCalculatedSideModel(), 'labeler__min_target': 0.006, 'labeler__ptsl_scalers': (1, 3), 'labeler__ewma_window': 10, 'labeler__num_days_exit': 50, 'enable_sample_weight': True, 'feature_engineer__sampler': EventSampler(cols=['close', 'balance_1k_native', 'net_flow_native', 'vix_close', 'spy_close']), 'feature_engineer__post_labeling_primary_model': TreeModel(model=DecisionTreeClassifier(class_weight='balanced', min_weight_fraction_leaf=0.2),cols=['close', 'balance_10k_native', 'balance_100k_native', 'balance_1k_native', 'balance_1k_usd', 'balance_10k_usd', 'balance_100k_usd', 'balance_1m_usd', 'balance_10m_usd', 'net_flow_native', 'net_flow_usd', 'vix_close', 'spy_close'])}
# Score :  0.8128723460145272
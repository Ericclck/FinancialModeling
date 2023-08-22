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
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from utils.sequential_bootstrapping import *


df = get_data("TMF")

X_pipe = Pipeline([
    ('ImproveSkewness', Fracdiff(0.7)),
    ('OutlierSmoother',None),
    ('Scaler',StandardScaler()),
    ('DimensionReduction',None),
    ('ToDf',ToDf()),
])

wrapper = ClassifierWrapper(cusum_threshold=0.2,X_pipe=X_pipe,ptsl_scalers=(2,2),model=RandomForestClassifier(),min_target=0.001,ewma_window=10,primary_model=crossing_ma((10,20)),sample_weight=False,num_days_exit=10)
wrapper.set_params(**{'model': GradientBoostingClassifier(max_depth=10), 'model__n_estimators': 1000, 'model__max_depth': 10, 'X_pipe__ImproveSkewness': Fracdiff(d=0.7, window=10), 'cusum_threshold': 0.4, 'primary_model': crossing_ma(fast_slow=(20, 40)), 'min_target': 0.002, 'ptsl_scalers': (3, 1), 'ewma_window': 20})


wrapper.fit(df)

print(get_average_uniqueness(construct_indicator_matrix(df.index,wrapper.labeler.first_touch_time)).mean())
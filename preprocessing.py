from utils.bars import get_events
from utils.get_data import get_filtered_data
import pandas as pd
import numpy as np
from utils.plot import plot_kde
from utils.custom_class import OutlierSmoother,ToDf
from sklearn.pipeline import Pipeline

df_filtered = get_filtered_data("TMF")

print(df_filtered.skew())

# fractional differentiation 
from fracdiff.sklearn import FracdiffStat,Fracdiff
from sklearn.preprocessing import StandardScaler
f =  Fracdiff(0.7)
X = f.fit_transform(df_filtered)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X, index=df_filtered.index, columns=df_filtered.columns)

print(X.skew())

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
X = pt.fit_transform(X)
X = pd.DataFrame(X, index=df_filtered.index, columns=df_filtered.columns)

print(X.skew())

pipe = Pipeline([
    ('improve_skewness', None),
    ('OutlierSmoother',OutlierSmoother([0.01,0.99])),
    ('Scaler',StandardScaler()),
    ('ToDf',ToDf(df_filtered.columns,df_filtered.index)),
])

pipe.set_params(**{'improve_skewness':Fracdiff(0.7)})

df_normalized = pipe.fit_transform(df_filtered)

plot_kde(df_normalized)





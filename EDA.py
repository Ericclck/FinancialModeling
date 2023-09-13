from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
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
from statsmodels.graphics.tsaplots import plot_acf

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

df = df.iloc[:int(len(df)*0.8)]

# non-stationary plots
# df[["close","vix_close","spy_close"]].plot(subplots=True,figsize=(10,10),title="Price plots")
# plt.show()

# df[['balance_1k_native','balance_10k_native','balance_100k_native']].plot(subplots=True,figsize=(10,10),title="Balance in native units plots")
# plt.show()

# df[['balance_1k_usd','balance_10k_usd','balance_100k_usd','balance_1m_usd','balance_10m_usd']].plot(figsize=(10,10),title="Balance in USD plots")
# plt.show()

df[['net_flow_native']].plot(subplots=True,figsize=(10,10),title="Net flow plots")
plt.show()

# Calculate returns
df['return'] = df['close'].pct_change().shift(-1)

balance_1k_native_signal = CumulativeSumSignaler(5,event_col="balance_1k_native").get_event(df)
net_flow_native_signal = CumulativeSumSignaler(5000,event_col="net_flow_native").get_event(df)
balance_1k_usd_signal = CumulativeSumSignaler(100000,event_col="balance_1k_usd").get_event(df)
close_signal = CumulativeSumSignaler(1000,event_col="close").get_event(df)
spy_close_signal = CumulativeSumSignaler(5,event_col="spy_close").get_event(df)
vix_close_signal = CumulativeSumSignaler(1,event_col="vix_close").get_event(df)
balance_10k_usd_signal = CumulativeSumSignaler(50000,event_col="balance_10k_usd").get_event(df)
balance_100k_usd_signal = CumulativeSumSignaler(5000,event_col="balance_100k_usd").get_event(df)
balance_1m_usd_signal = CumulativeSumSignaler(500,event_col="balance_1m_usd").get_event(df)
balance_10m_usd_signal = CumulativeSumSignaler(100,event_col="balance_10m_usd").get_event(df)

ax1 = plt.subplot(431)
plot_acf(df.close.pct_change().dropna(), lags=10, zero=False, ax=ax1)
ax1.set_ylim(-0.15,0.15)
plt.title('BTC return without filter')

ax2 = plt.subplot(432)
plot_acf(df.loc[balance_1k_native_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax2)
ax2.set_ylim(-0.15,0.15)
plt.title('BTC return with balance_1k_native filter')

ax3 = plt.subplot(433)
plot_acf(df.loc[net_flow_native_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax3)
ax3.set_ylim(-0.15,0.15)
plt.title('BTC return with net_flow_native filter')

ax4 = plt.subplot(434)
plot_acf(df.loc[balance_1k_usd_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax4)
ax4.set_ylim(-0.2,0.2)
plt.title('BTC return with balance_1k_usd filter')

ax5 = plt.subplot(435)
plot_acf(df.loc[close_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax5)
ax5.set_ylim(-0.15,0.15)
plt.title('BTC return with close filter')

ax6 = plt.subplot(436)
plot_acf(df.loc[spy_close_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax6)
ax6.set_ylim(-0.15,0.15)
plt.title('BTC return with spy_close filter')

ax7 = plt.subplot(437)
plot_acf(df.loc[vix_close_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax7)
ax7.set_ylim(-0.15,0.15)
plt.title('BTC return with vix_close filter')

ax8 = plt.subplot(438)
plot_acf(df.loc[balance_10k_usd_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax8)
ax8.set_ylim(-0.15,0.15)
plt.title('BTC return with balance_10k_usd filter')

ax9 = plt.subplot(439)
plot_acf(df.loc[balance_100k_usd_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax9)
ax9.set_ylim(-0.15,0.15)
plt.title('BTC return with balance_100k_usd filter')

ax10 = plt.subplot(4,3,10)
plot_acf(df.loc[balance_1m_usd_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax10)
ax10.set_ylim(-0.15,0.15)
plt.title('BTC return with balance_1m_usd filter')

ax11 = plt.subplot(4,3,11)
plot_acf(df.loc[balance_10m_usd_signal,'close'].pct_change().dropna(), lags=10, zero=False, ax=ax11)
ax11.set_ylim(-0.15,0.15)
plt.title('BTC return with balance_10m_usd filter')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()



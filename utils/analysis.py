import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
import statsmodels.api as sm


def resample_and_plot(df, freq='h'):
    """
    Resample a DataFrame by timestamp and plot the counts.
    
    df: Pandas DataFrame with a 'time' column of timestamps
    freq: Resampling frequency - 'h' for hour, 'm' for minute
    """
    # Resample by hour or minute and count rows


    time_bars = pd.to_datetime(df['time']).resample(freq, label='right').count()
    
    # Plot the result
    time_bars.plot(kind='bar')
    plt.xlabel('Time (hrs)')
    plt.ylabel('Count')
    plt.title(f'Count by {freq}')
    plt.show()

def ks_normality_test(data):
    """
    Performs the Kolmogorov-Smirnov test for normality and returns the p-value.
    """
    statistic, p_value = stats.kstest(data, 'norm')
    print('Kolmogorov-Smirnov Statistic=%.3f, p=%.3f' % (statistic, p_value))
    return p_value


def normal_plots(prices : pd.Series,title : str, log_diff=False):
    if log_diff:
        log_returns = np.log(prices).diff().dropna()
    else:
        log_returns = prices
    plt.figure(figsize=(6,12))
    plt.subplot(121)
    stats.probplot(log_returns, plot=plt)
    plt.axvline(-3)
    plt.axvline(3.2)
    plt.title(title + " QQ Plot")
    # kde plot
    plt.subplot(122)
    sns.kdeplot((log_returns-log_returns.mean())/(log_returns.std()), label="Dollar", linewidth=2, color='darkcyan')
    sns.kdeplot(np.random.normal(size=1000000), label="Normal", color='black', linestyle="--")
    plt.title(title + " KDE Plot " + f"KS-test pvalue : {ks_normality_test(log_returns):.5f}" + f" jb_test pvalue : {jarque_bera_normality_test(log_returns)[1]:.5f}")
    plt.show()

from statsmodels.tsa.stattools import adfuller, kpss

def adf_tests(series,log_diff=False, print_results=True):
    """
    Performs ADF and KPSS tests for stationarity and returns p-values.
    series: Pandas series
    print_results: If True, prints the results
                    If False, returns p-values without printing 
    """
    if log_diff:
        series = np.log(series).diff().dropna()
    # ADF test
    adf_result = adfuller(series)
    adf_pvalue = adf_result[1]
    
    
    # Print results
    if print_results:
        print('ADF p-value: %f' % adf_pvalue)
        
    # Return p-values
    return adf_pvalue

from scipy.stats import jarque_bera

def jarque_bera_normality_test(series, significance_level=0.05):
    # Perform the Jarque-Bera test
    jb_stat, p_value = jarque_bera(series)

    # Print the test result
    print(f"Jarque-Bera statistic: {jb_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Test for normality
    if p_value < significance_level:
        print("The series is not normally distributed.")
    else:
        print("The series is normally distributed.")

    # Return the test result (statistic and p-value)
    return jb_stat, p_value

def perform_dw_test(df):
    dw_results = pd.DataFrame(index=df.columns, columns=["DW Statistic"])
    for col in df.columns:
        dw_results.loc[col, "DW Statistic"] = sm.stats.durbin_watson(df[col])
    return dw_results
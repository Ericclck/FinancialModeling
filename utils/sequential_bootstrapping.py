import pandas as pd
import numpy as np

def construct_indicator_matrix(price_indices : pd.Index, first_touch_index:pd.Series) -> pd.DataFrame:
    """
    Construct indicator matrix that signifies whether a price bar is within the observation period of each sample

    Parameters
    ----------
    price_indices : pd.Series
        Series of indices of prices
    first_touch_index : pd.DataFrame
        DataFrame of first_touch_index
        index represents the start of the observation period
        first_touch_index column represents the timestamp of when the vertical barrier is reached

    Returns
    -------
    pd.DataFrame
        DataFrame of indicator matrix
        index represents price bar
        columns represent the samples
    """

    indicator_matrix = pd.DataFrame(0, index=price_indices, columns=range(first_touch_index.shape[0]))
    for i, (start, end) in enumerate(zip(first_touch_index.index, first_touch_index)):
        indicator_matrix.loc[start:end, i] = 1
    return indicator_matrix

def get_average_uniqueness(indicator_matrix : pd.DataFrame) -> pd.Series:
    """
    Get average uniqueness series (per sample)

    Parameters
    ----------
    indicator_matrix : pd.DataFrame
        DataFrame of indicator matrix
        index represents price bar
        columns represent the samples
    
    Returns
    -------
    pd.Series
        Series of average uniqueness (per sample)
    """

    # Get number of concurrency per price bar
    concurrency_per_price_bar = indicator_matrix.sum(axis=1)
    # Get uniqueness 
    uniqueness = indicator_matrix.div(concurrency_per_price_bar,axis=0)
    # Get average uniqueness per sample
    uniqueness = uniqueness[uniqueness>0]
    average_uniqueness = uniqueness.mean(axis=0)
    return average_uniqueness

def get_sequential_bootstrapping(indicator_matrix : pd.DataFrame,bootstrapping_length : int = None) -> list[int]:
    """
    Sequential bootstrapping

    Parameters
    ----------
    indicator_matrix : pd.DataFrame
        DataFrame of indicator matrix
        index represents price bar
        columns represent the samples
    bootstrapping_length : int
        Length of bootstrapping
    
    Returns
    -------
    list(int)
        List of indices of selected samples
    """
    if bootstrapping_length == None: 
        bootstrapping_length = indicator_matrix.shape[1]
    selected_indices = []
    while len(selected_indices) < bootstrapping_length:
        reduced_uniqueness = pd.Series()
        for col in indicator_matrix.columns:
            reduced_indicator_matrix = indicator_matrix.loc[:,selected_indices + [col]]
            reduced_uniqueness[col] = get_average_uniqueness(reduced_indicator_matrix).iloc[-1]
        prob = reduced_uniqueness / reduced_uniqueness.sum()
        selected_indices.append(np.random.choice(indicator_matrix.columns,p=prob))
    return selected_indices
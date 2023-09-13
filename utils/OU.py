import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm

def OU_likelihood(params, data):
    theta, mu, sigma = params
    dt = np.diff(data.index)  # Assumes data has a DateTime index
    X_diff = np.diff(data.values.squeeze())
    
    drift = theta * (mu - data.values[:-1]) * dt
    diffusion = sigma * np.sqrt(dt)
    
    return -np.sum(norm.logpdf(X_diff, loc=drift, scale=diffusion))
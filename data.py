from utils.get_data import get_ticker_data
import os

data = get_ticker_data("SPY")

data.to_csv(os.path.join(os.path.dirname(__file__), "data", "SPY.csv"),index=False)
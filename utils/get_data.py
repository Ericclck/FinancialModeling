import yfinance as yf
import pandas as pd

def store_ticker_data(ticker:str):
    t = yf.Ticker(ticker)
    
    # Get historical market data
    hist = t.history(period="max")

    # Convert to pandas DataFrame
    hist_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    
    # Keep only 'Date' and OHLC columns
    hist_df = hist_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    hist_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Save to csv
    hist_df.to_csv(f'data/{ticker}.csv', index=False)

def get_data(ticker:str) -> pd.Series:
    return pd.read_csv(f'data/{ticker}.csv', index_col="date", parse_dates=True)

def get_updated_data(ticker:str) -> pd.Series:
    store_ticker_data(ticker)
    return get_data(ticker)
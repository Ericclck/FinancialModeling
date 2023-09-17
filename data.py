import requests
import pandas as pd

def extract_sentiment_and_time(data):
    # Initialize lists to hold sentiment scores and time published
    sentiment_scores = []
    time_published = []
    
    # Loop through each item in the feed
    for item in data['feed']:
        # Append the sentiment score and time published to respective lists
        sentiment_scores.append(item['overall_sentiment_score'])
        time_published.append(item['time_published'])
    
    # Create a DataFrame using these lists
    df = pd.DataFrame({
        'time_published': time_published,
        'overall_sentiment_score': sentiment_scores
    })
    
    return df

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=FOREX:USD&topics=economy_monetary&apikey=VTHB8AYKXDG0O6GG'
r = requests.get(url)
data = r.json()

import json
# Save JSON data
with open('data/SPY_news_sentiment.json', 'w') as outfile:
    json.dump(data, outfile)

# convert json to pandas dataframe then save to data/
df = extract_sentiment_and_time(data)
df.to_csv('data/SPY_news_sentiment.csv', index=False)
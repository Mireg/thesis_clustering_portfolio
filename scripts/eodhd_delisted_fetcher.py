import requests
import pandas as pd
from dotenv import load_dotenv
import json
import os

load_dotenv()
API_KEY = os.environ.get('EODHD_API_KEY')

start_date = "2010-01-01"
end_date = "2025-01-01"

with open('data/clusters/cluster_assignments_5_2bf0de2051e663b248d2.table.json', 'r') as f:
    json_data = json.load(f)

clusters = pd.DataFrame(json_data['data'], columns=json_data['columns'])
tickers = clusters['ticker'].tolist()

for ticker in tickers:
    # Handling of Berkshire as a special case
    if ticker == 'BRK.B':
        ticker = 'BRK'
    url = f'https://eodhd.com/api/eod/{ticker}?from={start_date}&to={end_date}&api_token={API_KEY}&fmt=json&period=m'
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df.to_csv(f'data/stock_prices/{ticker}.csv')
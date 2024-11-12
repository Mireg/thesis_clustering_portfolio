import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.environ.get("QUANDL_KEY")

#def get_financial_data(tickers, start_date="2010-01-01"):
#    financial_data = []
    
#    start_date = pd.to_datetime(start_date)
    
#    for ticker in tickers:


#tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  

##financials_data.to_csv('Data/financials_test.csv')
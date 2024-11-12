import requests
from bs4 import BeautifulSoup
import pandas as pd

#change the url if you want to get historical list
#Current list: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
#2010:  http://web.archive.org/web/20100210015201/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
#2015: 
#2017:
#2019: http://web.archive.org/web/20190107004442/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_sp500_tickers(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'class': 'wikitable'})

    tickers = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) > 1:
            ticker = cols[0].text.strip()
            tickers.append(ticker)
    
    return tickers

tickers = get_sp500_tickers(url)

print(tickers[:10])

df = pd.DataFrame(tickers, columns=["Ticker"])
df.to_csv("sp500_tickers.csv", index=False)

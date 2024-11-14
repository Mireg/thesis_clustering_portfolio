import pandas as pd
import requests
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

class SECEdgarFetcher:
    def __init__(self, cik_map_file="data/ticker_to_cik.csv"):
        self.base_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
        self.headers = {
            'User-Agent': str(os.environ.get('SEC_AGENT')),
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        self.cik_map = pd.read_csv(cik_map_file, dtype={'cik': str}).set_index('ticker')['cik'].to_dict()
        self.raw_data = [] 

    def get_financial_statements(self, ticker, output_dir="sec_data/json"):
        """Fetch financial statements for a given ticker and save the raw JSON."""
        if ticker not in self.cik_map:
            print(f"Could not find CIK for ticker {ticker}")
            return None

        cik = self.cik_map[ticker]
        url = self.base_url.format(cik)
        
        try:
            print(f"Fetching data from URL: {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            # Save raw JSON data
            os.makedirs(output_dir, exist_ok=True)
            json_file = os.path.join(output_dir, f"{ticker}.json")
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved raw JSON for {ticker} to {json_file}")
            
            # Extract financial statement data
            facts = data.get('facts', {})
            us_gaap = facts.get('us-gaap', {})
            
            # Key financial metrics to extract
            metric_tag_mapping = {
                'Revenue': [
                    'Revenues',
                    'RevenueFromContractWithCustomerExcludingAssessedTax',
                    'SalesRevenueGoodsNet',
                    'SalesRevenueNet'
                ],
                
                'OperatingIncome': [
                    'OperatingIncomeLoss',
                    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
                ],
                
                'GrossProfit': [
                    'GrossProfit'
                ],
                
                'ResearchAndDevelopment': [
                    'ResearchAndDevelopmentExpense'
                ],
                
                'OperatingExpenses': [
                    'OperatingExpenses',
                    'OperatingCostsAndExpenses'
                ],
                
                'NetIncomeLoss': [
                    'NetIncomeLoss',
                    'ProfitLoss'
                ],
                
                'EPS': [
                    'EarningsPerShareBasic',
                    'EarningsPerShareDiluted'
                ],
                
                'SellingGeneral': [
                    'SellingGeneralAndAdministrativeExpense'
                ],
                
                'CostOfRevenue': [
                    'CostOfGoodsAndServicesSold',
                    'CostOfRevenue',
                    'CostOfServices'
                ],

                'Restructuring': [
                    'RestructuringCharges',
                    'RestructuringCosts'
                ],

                'OperatingExpense': [
                    'OperatingExpenses',
                    'GeneralAndAdministrativeExpense',
                    'SellingAndMarketingExpense'
                ]
            }
            
            ticker_data = []
            
            for metric, tags in metric_tag_mapping.items():
                for tag in tags:
                    if tag in us_gaap:
                        units = us_gaap[tag].get('units', {})
                        if 'USD' in units:
                            for record in units['USD']:
                                if record.get('form') in ['10-Q', '10-K']:
                                    ticker_data.append({
                                        'ticker': ticker,
                                        'metric': metric,  # Use the standard metric name
                                        'date': record.get('end'),
                                        'value': record.get('val')
                                    })
                        break
            
            return pd.DataFrame(ticker_data)
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def fetch_all(self, tickers, output_dir="Data/sec_data"):
        """Fetch and consolidate data for all tickers and save raw data."""
        os.makedirs(output_dir, exist_ok=True)
        
        for ticker in tickers:
            print(f"\nFetching data for {ticker}")
            ticker_df = self.get_financial_statements(ticker, output_dir=os.path.join(output_dir, "json"))
            if ticker_df is not None and not ticker_df.empty:
                self.raw_data.append(ticker_df)
            time.sleep(2)  # Be nice to SEC servers
        
        if self.raw_data:
            # Save raw data to CSV
            raw_data_df = pd.concat(self.raw_data, ignore_index=True)
            raw_data_file = os.path.join(output_dir, "raw_financial_data.csv")
            raw_data_df.to_csv(raw_data_file, index=False)
            print(f"Raw data saved to {raw_data_file}")
            return raw_data_df
        else:
            print("No data fetched.")
            return pd.DataFrame()

    def pivot_data(self, df, output_dir="Data/sec_data"):
        """Pivot the data into MultiIndex format with ticker and date, and save the result."""
        if df.empty:
            print("No data to process.")
            return None
        
        # Pivot the data
        pivoted = df.pivot_table(
            index=['ticker', 'date'],  # MultiIndex
            columns='metric',
            values='value',
            aggfunc='mean'
        )
        
        # Flatten the column MultiIndex (optional)
        pivoted.columns = [col for col in pivoted.columns]
        
        # Save pivoted data to CSV
        pivoted_file = os.path.join(output_dir, "pivoted_financial_data.csv")
        pivoted.to_csv(pivoted_file)
        print(f"Pivoted data saved to {pivoted_file}")
        
        return pivoted


def main():
    fetcher = SECEdgarFetcher()

    # List of companies to fetch
    #tickers = ['AAPL', 'MSFT', 'GOOGL']
    tickers = pd.read_csv('Data/sp500_tickers.csv')['Ticker'].tolist()
    
    # Step 1: Fetch data for all tickers and save raw data
    raw_data = fetcher.fetch_all(tickers)
    
    # Step 2: Pivot the data and save the pivoted DataFrame
    pivoted_df = fetcher.pivot_data(raw_data)
    
    if pivoted_df is not None:
        print("\nPivoted Data (MultiIndex with Ticker and Date):")
        print(pivoted_df.head())


if __name__ == "__main__":
    main()

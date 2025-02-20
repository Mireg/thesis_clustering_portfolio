import pandas as pd
import requests
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

class SECEdgarFetcher:
    def __init__(self, cik_map_file="./Data/company_tickers_2010_full.csv"):
        self.base_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
        self.headers = {
            'User-Agent': str(os.environ.get('SEC_AGENT')),
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        self.cik_map = pd.read_csv(cik_map_file, dtype={'cik': str}).set_index('ticker')['cik'].map(lambda x: x.zfill(10)).to_dict()
        self.raw_data = []

    def get_financial_statements(self, ticker, output_dir="./Data/sec_data/json"):
        if ticker not in self.cik_map:
            print(f"Could not find CIK for ticker {ticker}")
            parent_dir = os.path.dirname(output_dir)
            os.makedirs(parent_dir, exist_ok=True)
            missing_ciks_file = os.path.join(parent_dir, "missing_ciks.csv")
            
            if not os.path.exists(missing_ciks_file):
                with open(missing_ciks_file, "w") as f:
                    f.write("ticker\n")
                    
            with open(missing_ciks_file, "a") as f:
                f.write(f"{ticker}\n")
            return None

        cik = self.cik_map[ticker]
        url = self.base_url.format(cik)
        
        try:
            time.sleep(0.2)  # SEC rate limit
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            os.makedirs(output_dir, exist_ok=True)
            json_file = os.path.join(output_dir, f"{ticker}.json")
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            facts = data.get('facts', {})
            us_gaap = facts.get('us-gaap', {})
            
            metric_tag_mapping = {
                # Balance Sheet
                'Assets': ['Assets'],
                'Liabilities': ['LiabilitiesAndStockholdersEquity'],
                'StockholdersEquity': ['StockholdersEquity'],
                'Goodwill': ['Goodwill'],
                'PPE': ['PropertyPlantAndEquipmentNet'],
                'CurrentAssets': ['AssetsCurrent'],
                'CurrentLiabilities': ['LiabilitiesCurrent'],
                'LongTermDebt': ['LongTermDebt'],
                
                # Cash Flow
                'OperatingCashFlow': ['NetCashProvidedByUsedInOperatingActivities'],
                'FinancingCashFlow': ['NetCashProvidedByUsedInFinancingActivities'],
                'InvestingCashFlow': ['NetCashProvidedByUsedInInvestingActivities'],
                'CapEx': ['PaymentsToAcquirePropertyPlantAndEquipment'],
                
                # Income Statement
                'Revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'],
                'OperatingIncome': ['OperatingIncomeLoss'],
                'NetIncome': ['NetIncomeLoss', 'ProfitLoss'],
                'GrossProfit': ['GrossProfit'],
                'OperatingExpenses': ['OperatingExpenses', 'OperatingCostsAndExpenses'],
                'RnD': ['ResearchAndDevelopmentExpense'],
                'SGA': ['SellingGeneralAndAdministrativeExpense'],
                'COGS': ['CostOfGoodsAndServicesSold', 'CostOfRevenue'],
                'InterestExpense': ['InterestExpense'],
                'IncomeTax': ['IncomeTaxExpenseBenefit'],
                
                # Additional Metrics
                'ShareBasedComp': ['ShareBasedCompensation'],
                'Amortization': ['AmortizationOfIntangibleAssets'],
                'WeightedShares': ['WeightedAverageNumberOfSharesOutstandingBasic'],
                'DilutedShares': ['WeightedAverageNumberOfDilutedSharesOutstanding']
            }
            
            ticker_data = []
            
            for metric, tags in metric_tag_mapping.items():
                for tag in tags:
                    if tag in us_gaap:
                        units = us_gaap[tag].get('units', {})
                        if 'USD' in units or 'shares' in units:
                            unit_type = 'USD' if 'USD' in units else 'shares'
                            for record in units[unit_type]:
                                if record.get('form') in ['10-Q', '10-K']:
                                    ticker_data.append({
                                        'ticker': ticker,
                                        'metric': metric,
                                        'date': record.get('end'),
                                        'value': record.get('val'),
                                        'form': record.get('form'),
                                        'frame': record.get('frame', '')
                                    })
                        break
            
            return pd.DataFrame(ticker_data)
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def fetch_all(self, tickers, output_dir="Data/sec_data"):
        os.makedirs(output_dir, exist_ok=True)
        
        for ticker in tickers:
            print(f"\nFetching data for {ticker}")
            ticker_df = self.get_financial_statements(ticker, output_dir=os.path.join(output_dir, "json"))
            if ticker_df is not None and not ticker_df.empty:
                self.raw_data.append(ticker_df)
                
        if self.raw_data:
            raw_data_df = pd.concat(self.raw_data, ignore_index=True)
            raw_data_file = os.path.join(output_dir, "raw_financial_data.csv")
            raw_data_df.to_csv(raw_data_file, index=False)
            print(f"Raw data saved to {raw_data_file}")
            return raw_data_df
        else:
            print("No data fetched.")
            return pd.DataFrame()

    def pivot_data(self, df, output_dir="./Data/sec_data"):
        if df.empty:
            print("No data to process.")
            return None
        
        pivoted = df.pivot_table(
            index=['ticker', 'date'],
            columns='metric',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        pivoted_file = os.path.join(output_dir, "pivoted_financial_data.csv")
        pivoted.to_csv(pivoted_file, index=False)
        print(f"Pivoted data saved to {pivoted_file}")
        
        return pivoted

def main():
    tickers = pd.read_csv('./Data/sp500_tickers_2010.csv')['Ticker'].tolist()
    fetcher = SECEdgarFetcher()
    raw_data = fetcher.fetch_all(tickers)
    pivoted_df = fetcher.pivot_data(raw_data)

if __name__ == "__main__":
    main()
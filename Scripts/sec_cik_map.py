import pandas as pd
from sec_cik_mapper import StockMapper
from pathlib import Path

# Initialize a stock mapper instance
mapper = StockMapper()

# Get mapping from ticker to CIK
ticker_to_cik = mapper.ticker_to_cik

# Convert the dictionary to a pandas DataFrame and format CIKs with leading zeros
ticker_to_cik_df = pd.DataFrame(list(ticker_to_cik.items()), columns=['ticker', 'cik'])
ticker_to_cik_df['cik'] = ticker_to_cik_df['cik'].apply(lambda x: str(x).zfill(10))

# Save the DataFrame to a CSV file
csv_path = Path("data/ticker_to_cik.csv")
ticker_to_cik_df.to_csv(csv_path, index=False)

print(f"CIK mapping saved to {csv_path}")

A repository for my masters thesis regarding using clustering methods for portfolio selection

https://www.sec.gov/files/company_tickers.json

https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

Some assumptions: 
    Unit inconsistencies are assumed to differ by factors of one million
    Growth rates are calculated using the Compound Annual Growth Rate (CAGR) model
    Companies transitioning between profit and loss (sign changes) are excluded from growth rate calculations
    Outliers are defined using z-scores with threshold 3.5
    Missing data is primarily addressed through size-based imputation
    Extremely small values are handled with offset log transformations to prevent undefined logarithms
    Financial ratios are bounded to realistic ranges to prevent extreme values from distorting analysis"
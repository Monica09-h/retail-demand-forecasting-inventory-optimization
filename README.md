# Retail Demand Forecasting and Inventory Optimization

**Author:** Monica Hall  
**Program:** M.S. Analytics, Saint Louis University  
**Expected graduation:** December 2026

## Project overview

This project analyzes two years of real online-retail transactions to forecast product demand and translate demand variability into practical inventory recommendations. It demonstrates an end-to-end supply chain analytics workflow: data cleaning, exploratory analysis, time-series validation, demand forecasting, safety-stock calculation, reorder-point design, and executive dashboarding.

## Business questions

1. How does demand change by month and season?
2. Which consistently active products generate the most revenue?
3. Which forecasting method performs best on unseen weeks?
4. How much safety stock is appropriate under a defined service-level scenario?
5. At what inventory position should replenishment be triggered?

## Dataset

The project uses the [UCI Online Retail II dataset](https://archive.ics.uci.edu/ml/datasets/online%2Bretail%2BII), containing 1,067,371 transactions from a UK-based online retailer between December 2009 and December 2011.

Important fields include invoice number, stock code, product description, quantity, invoice date, unit price, customer ID, and country.

## Data preparation

- Combined the two annual Excel sheets.
- Standardized column names and data types.
- Separated cancellations and negative-quantity returns.
- Retained positive-quantity, positive-price merchandise transactions.
- Excluded non-merchandise charges and adjustments.
- Created weekly and monthly demand tables.
- Removed the final partial week from model training.

The cleaned data contained 1,036,960 valid merchandise-sale rows, 4,893 products, 43 countries, and approximately GBP 20.1 million in revenue.

## Forecasting approach

Five high-value products with at least 90 active weeks and 500 orders were selected. Three methods were evaluated on the final 13 complete weeks:

- Four-week moving average
- Seasonal naive forecast using a 52-week lag
- Random forest using lag, rolling-average, and calendar features

Mean absolute error (MAE) was the primary model-selection metric. The four-week moving average produced the lowest MAE for all five selected products. This result illustrates why forecast validation should guide model choice rather than model complexity.

## Inventory-policy scenario

The inventory recommendations use these explicit assumptions:

- Replenishment lead time: 2 weeks
- Cycle-service level: 95%
- Service factor: 1.645
- Demand variability estimated from the latest 26 complete weeks

Safety stock is calculated as:

`Safety Stock = z × Weekly Demand Standard Deviation × √Lead Time`

The reorder point is calculated as:

`Reorder Point = Average Weekly Demand × Lead Time + Safety Stock`

| Product | Safety stock | Reorder point | 8-week forecast |
|---|---:|---:|---:|
| Regency Cakestand 3 Tier | 149 | 465 | 1,428 |
| White Hanging Heart T-Light Holder | 818 | 1,915 | 5,714 |
| Party Bunting | 513 | 1,156 | 1,691 |
| Assorted Colour Bird Ornament | 1,661 | 3,213 | 8,594 |
| Chilli Lights | 511 | 924 | 4,323 |

## Key findings

- Revenue and unit demand rise sharply from September and peak in November.
- High-volume products also show substantial weekly demand variability.
- Inventory buffers should be differentiated by product rather than set through a single blanket rule.
- The Assorted Colour Bird Ornament requires the largest modeled buffer because it combines high average demand with high variability.
- Forecasts and reorder parameters should be refreshed weekly, especially before the September-November peak.

## Limitations

The source records transactions rather than actual inventory balances, stockouts, purchase orders, or supplier lead times. Therefore, safety-stock and reorder-point outputs are scenario recommendations based on stated assumptions; they are not observed company policies. Promotional and holiday features were also unavailable.

## Tools

- Python: pandas, NumPy, scikit-learn, Matplotlib
- Excel dashboard
- Power BI-ready CSV output tables

## Repository contents

- `01_prepare_data.py` - data ingestion and cleaning
- `02_forecast_inventory.py` - model validation, forecasting, and inventory calculations
- `monica_hall_supply_chain_analytics_portfolio.xlsx` - dashboard workbook
- `project_data/` - analysis and dashboard-ready output tables
- `project_charts/` - portfolio visuals
- `monica_hall_supply_chain_analytics_case_study.pdf` - recruiter-facing case study


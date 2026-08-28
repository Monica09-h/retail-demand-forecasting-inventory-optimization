from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "online_retail_II.xlsx"
OUTPUT = ROOT / "project_data"
OUTPUT.mkdir(exist_ok=True)


def load_workbook() -> pd.DataFrame:
    sheets = pd.read_excel(SOURCE, sheet_name=None)
    frames = []
    for sheet_name, frame in sheets.items():
        frame["SourceSheet"] = sheet_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = raw.rename(
        columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "Customer ID": "CustomerID",
        }
    ).copy()
    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["StockCode"] = data["StockCode"].astype("string").str.strip()
    data["Description"] = data["Description"].astype("string").str.strip()
    data["Country"] = data["Country"].astype("string").str.strip()
    data["IsCancellation"] = data["InvoiceNo"].str.upper().str.startswith("C")
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]

    returns = data.loc[data["IsCancellation"] | (data["Quantity"] < 0)].copy()

    non_merchandise_codes = {
        "ADJUST", "ADJUST2", "AMAZONFEE", "B", "BANK CHARGES", "C2",
        "CRUK", "D", "DCGSSBOY", "DCGSSGIRL", "DOT", "GIFT_0001_10",
        "GIFT_0001_20", "GIFT_0001_30", "GIFT_0001_40", "GIFT_0001_50",
        "M", "PADS", "POST", "S", "SP1002", "TEST001", "TEST002",
    }
    sales = data.loc[
        (~data["IsCancellation"])
        & (data["Quantity"] > 0)
        & (data["UnitPrice"] > 0)
        & data["Description"].notna()
        & (~data["StockCode"].str.upper().isin(non_merchandise_codes))
    ].copy()

    sales["Date"] = sales["InvoiceDate"].dt.date
    sales["Week"] = sales["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    sales["Month"] = sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    sales["Year"] = sales["InvoiceDate"].dt.year
    sales["YearMonth"] = sales["InvoiceDate"].dt.strftime("%Y-%m")
    return sales, returns


def create_outputs(sales: pd.DataFrame, returns: pd.DataFrame) -> None:
    sales.to_csv(OUTPUT / "clean_sales.csv.gz", index=False, compression="gzip")
    returns.to_csv(
        OUTPUT / "returns_and_cancellations.csv.gz", index=False, compression="gzip"
    )

    monthly = (
        sales.groupby(["Month", "StockCode", "Description", "Country"], as_index=False)
        .agg(UnitsSold=("Quantity", "sum"), Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
    )
    monthly.to_csv(OUTPUT / "monthly_product_demand.csv", index=False)

    weekly_uk = (
        sales.loc[sales["Country"].eq("United Kingdom")]
        .groupby(["Week", "StockCode", "Description"], as_index=False)
        .agg(UnitsSold=("Quantity", "sum"), Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
    )
    weekly_uk.to_csv(OUTPUT / "weekly_uk_product_demand.csv", index=False)

    product_summary = (
        sales.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            UnitsSold=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            ActiveWeeks=("Week", "nunique"),
            FirstSale=("InvoiceDate", "min"),
            LastSale=("InvoiceDate", "max"),
        )
        .sort_values("Revenue", ascending=False)
    )
    product_summary.to_csv(OUTPUT / "product_summary.csv", index=False)


def main() -> None:
    raw = load_workbook()
    sales, returns = prepare_transactions(raw)
    create_outputs(sales, returns)
    print(f"Raw rows: {len(raw):,}")
    print(f"Valid merchandise sales: {len(sales):,}")
    print(f"Returns/cancellations: {len(returns):,}")
    print(f"Sales period: {sales['InvoiceDate'].min()} to {sales['InvoiceDate'].max()}")
    print(f"Products: {sales['StockCode'].nunique():,}")
    print(f"Countries: {sales['Country'].nunique():,}")
    print(f"Revenue: GBP {sales['Revenue'].sum():,.2f}")


if __name__ == "__main__":
    main()

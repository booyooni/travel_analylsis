from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def get_default_data_path(app_dir: Path) -> Path:
    return app_dir / "data" / "online_retail.parquet"


def prepare_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Required columns are missing from dataset: {missing_cols}")

    prepared = df.copy()
    prepared["InvoiceDate"] = pd.to_datetime(prepared["InvoiceDate"])
    prepared["InvoiceNoStr"] = prepared["InvoiceNo"].astype(str)
    prepared["StockCodeStr"] = prepared["StockCode"].astype(str)
    prepared["DescriptionStr"] = prepared["Description"].astype("string").fillna("Unknown")
    prepared["CountryStr"] = prepared["Country"].astype("string").fillna("Unknown")
    prepared["CustomerIDStr"] = (
        prepared["CustomerID"]
        .astype("Int64")
        .astype("string")
        .fillna("")
    )
    prepared["IsCancelled"] = prepared["InvoiceNoStr"].str.startswith("C")
    prepared["Sales"] = prepared["Quantity"] * prepared["UnitPrice"]
    prepared["OrderDate"] = prepared["InvoiceDate"].dt.date
    prepared["YearMonth"] = prepared["InvoiceDate"].dt.to_period("M").astype(str)
    prepared["YearMonthStart"] = prepared["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    return prepared


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (df["CustomerIDStr"] != "")
        & (df["DescriptionStr"] != "Unknown")
    ].copy()
    return clean


def load_online_retail(data_path: Path) -> dict[str, pd.DataFrame | dict[str, object]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    raw = pd.read_parquet(data_path)
    prepared = prepare_online_retail(raw)
    clean = build_clean_dataset(prepared)

    metadata = {
        "data_path": str(data_path),
        "raw_rows": int(len(prepared)),
        "clean_rows": int(len(clean)),
        "raw_orders": int(prepared["InvoiceNoStr"].nunique()),
        "clean_orders": int(clean["InvoiceNoStr"].nunique()),
        "raw_customers": int(prepared.loc[prepared["CustomerIDStr"] != "", "CustomerIDStr"].nunique()),
        "clean_customers": int(clean["CustomerIDStr"].nunique()),
        "raw_products": int(prepared["StockCodeStr"].nunique()),
        "clean_products": int(clean["StockCodeStr"].nunique()),
    }
    return {"raw": prepared, "clean": clean, "metadata": metadata}


def filter_transactions(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    countries: list[str] | None = None,
    product_query: str = "",
    customer_query: str = "",
    include_cancellations: bool = False,
) -> pd.DataFrame:
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    filtered = df[(df["OrderDate"] >= start_date) & (df["OrderDate"] <= end_date)].copy()

    if countries:
        filtered = filtered[filtered["CountryStr"].isin(countries)]
    if product_query:
        query = product_query.strip().lower()
        if query:
            filtered = filtered[
                filtered["StockCodeStr"].str.lower().str.contains(query, na=False, regex=False)
                | filtered["DescriptionStr"].str.lower().str.contains(query, na=False, regex=False)
            ]
    if customer_query:
        query = customer_query.strip().lower()
        if query:
            filtered = filtered[filtered["CustomerIDStr"].str.lower().str.contains(query, na=False, regex=False)]
    if not include_cancellations:
        filtered = filtered[~filtered["IsCancelled"]]

    return filtered

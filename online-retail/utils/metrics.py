from __future__ import annotations

import pandas as pd


def _most_frequent_or_first(series: pd.Series) -> str:
    mode = series.mode()
    if not mode.empty:
        return str(mode.iat[0])
    return str(series.iloc[0])


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "rows": 0,
            "orders": 0,
            "products": 0,
            "customers": 0,
            "sales": 0.0,
            "aov": 0.0,
            "arppu": 0.0,
            "cancelled_orders": 0,
            "cancel_rate": 0.0,
            "new_customers": 0,
        }

    orders = int(df["InvoiceNoStr"].nunique())
    customers_df = df[df["CustomerIDStr"] != ""]
    customers = int(customers_df["CustomerIDStr"].nunique())
    sales = float(df["Sales"].sum())
    cancelled_orders = int(df.loc[df["IsCancelled"], "InvoiceNoStr"].nunique())
    first_month = customers_df.groupby("CustomerIDStr")["YearMonthStart"].min()
    current_month = df["YearMonthStart"].max()
    new_customers = int((first_month == current_month).sum()) if not first_month.empty else 0

    return {
        "rows": int(len(df)),
        "orders": orders,
        "products": int(df["StockCodeStr"].nunique()),
        "customers": customers,
        "sales": sales,
        "aov": sales / orders if orders else 0.0,
        "arppu": sales / customers if customers else 0.0,
        "cancelled_orders": cancelled_orders,
        "cancel_rate": cancelled_orders / orders if orders else 0.0,
        "new_customers": new_customers,
    }


def aggregate_countries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["CountryStr", "sales", "orders", "customers", "aov"])
    grouped = (
        df.groupby("CountryStr", as_index=False)
        .agg(
            sales=("Sales", "sum"),
            orders=("InvoiceNoStr", "nunique"),
            customers=("CustomerIDStr", lambda s: s[s != ""].nunique()),
        )
        .sort_values("sales", ascending=False)
    )
    grouped["aov"] = grouped["sales"] / grouped["orders"].replace(0, pd.NA)
    grouped["aov"] = grouped["aov"].fillna(0.0)
    return grouped


def aggregate_products(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["DescriptionStr"] != "Unknown"].copy()
    if base.empty:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "sales", "quantity", "orders", "customers"])
    grouped = (
        base.groupby("StockCodeStr", as_index=False)
        .agg(
            DescriptionStr=("DescriptionStr", _most_frequent_or_first),
            sales=("Sales", "sum"),
            quantity=("Quantity", "sum"),
            orders=("InvoiceNoStr", "nunique"),
            customers=("CustomerIDStr", lambda s: s[s != ""].nunique()),
        )
        .sort_values(["sales", "orders", "quantity"], ascending=False)
    )
    return grouped


def aggregate_customers(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["CustomerIDStr"] != ""].copy()
    if base.empty:
        return pd.DataFrame(columns=["CustomerIDStr", "sales", "orders", "avg_order_value", "last_purchase", "country"])

    grouped = (
        base.groupby("CustomerIDStr", as_index=False)
        .agg(
            sales=("Sales", "sum"),
            orders=("InvoiceNoStr", "nunique"),
            last_purchase=("InvoiceDate", "max"),
            country=("CountryStr", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
        .sort_values("sales", ascending=False)
    )
    grouped["avg_order_value"] = grouped["sales"] / grouped["orders"].replace(0, pd.NA)
    grouped["avg_order_value"] = grouped["avg_order_value"].fillna(0.0)
    return grouped


def aggregate_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["InvoiceNoStr", "InvoiceDate", "CustomerIDStr", "CountryStr", "line_count", "quantity", "sales", "is_cancelled"])

    grouped = (
        df.groupby("InvoiceNoStr", as_index=False)
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            CustomerIDStr=("CustomerIDStr", lambda s: next((v for v in s if v), "")),
            CountryStr=("CountryStr", lambda s: s.iloc[0]),
            line_count=("InvoiceNoStr", "size"),
            quantity=("Quantity", "sum"),
            sales=("Sales", "sum"),
            is_cancelled=("IsCancelled", "max"),
        )
        .sort_values("InvoiceDate", ascending=False)
    )
    return grouped


def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["CustomerIDStr"] != ""].copy()
    if base.empty:
        return pd.DataFrame(columns=["OrderDate", "buyers", "orders", "sales"])
    return (
        base.groupby("OrderDate", as_index=False)
        .agg(buyers=("CustomerIDStr", "nunique"), orders=("InvoiceNoStr", "nunique"), sales=("Sales", "sum"))
        .sort_values("OrderDate")
    )


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    if base.empty:
        return pd.DataFrame(columns=["YearMonth", "YearMonthStart", "sales", "orders", "buyers"])
    return (
        base.groupby(["YearMonth", "YearMonthStart"], as_index=False)
        .agg(sales=("Sales", "sum"), orders=("InvoiceNoStr", "nunique"), buyers=("CustomerIDStr", lambda s: s[s != ""].nunique()))
        .sort_values("YearMonthStart")
    )


def monthly_business_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["CustomerIDStr"] != ""].copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "YearMonth",
                "YearMonthStart",
                "sales",
                "orders",
                "buyers",
                "arppu",
                "aov",
                "acquired_customers",
                "repeat_customers",
                "repeat_rate",
            ]
        )

    monthly = monthly_summary(base)
    first_month = base.groupby("CustomerIDStr")["YearMonthStart"].min().rename("first_month")
    base = base.merge(first_month, on="CustomerIDStr", how="left")

    acquired = (
        base[base["YearMonthStart"] == base["first_month"]]
        .groupby(["YearMonth", "YearMonthStart"], as_index=False)
        .agg(acquired_customers=("CustomerIDStr", "nunique"))
    )
    repeat = (
        base[base["YearMonthStart"] > base["first_month"]]
        .groupby(["YearMonth", "YearMonthStart"], as_index=False)
        .agg(repeat_customers=("CustomerIDStr", "nunique"))
    )

    merged = monthly.merge(acquired, on=["YearMonth", "YearMonthStart"], how="left")
    merged = merged.merge(repeat, on=["YearMonth", "YearMonthStart"], how="left")
    merged[["acquired_customers", "repeat_customers"]] = merged[["acquired_customers", "repeat_customers"]].fillna(0)
    merged["arppu"] = merged["sales"] / merged["buyers"].replace(0, pd.NA)
    merged["aov"] = merged["sales"] / merged["orders"].replace(0, pd.NA)
    merged["repeat_rate"] = merged["repeat_customers"] / merged["buyers"].replace(0, pd.NA)
    return merged.fillna(0)


def customer_monthly_sales(df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    base = df[df["CustomerIDStr"] == customer_id].copy()
    if base.empty:
        return pd.DataFrame(columns=["YearMonth", "YearMonthStart", "sales", "orders"])
    return (
        base.groupby(["YearMonth", "YearMonthStart"], as_index=False)
        .agg(sales=("Sales", "sum"), orders=("InvoiceNoStr", "nunique"))
        .sort_values("YearMonthStart")
    )


def product_monthly_sales(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    base = df[df["StockCodeStr"] == stock_code].copy()
    if base.empty:
        return pd.DataFrame(columns=["YearMonth", "YearMonthStart", "sales", "quantity", "orders"])
    return (
        base.groupby(["YearMonth", "YearMonthStart"], as_index=False)
        .agg(sales=("Sales", "sum"), quantity=("Quantity", "sum"), orders=("InvoiceNoStr", "nunique"))
        .sort_values("YearMonthStart")
    )

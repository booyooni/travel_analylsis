from __future__ import annotations

import pandas as pd


def _score_series(series: pd.Series, ascending: bool) -> pd.Series:
    ranked = series.rank(method="first", ascending=ascending)
    bucket_count = int(min(5, max(1, ranked.nunique())))
    labels = list(range(1, bucket_count + 1))
    scored = pd.qcut(ranked, bucket_count, labels=labels, duplicates="drop")
    scored = scored.astype(float).fillna(1).astype(int)
    if bucket_count < 5:
        # Expand to the 1-5 scale so downstream segment rules stay consistent.
        scored = (((scored - 1) * 4) / max(1, bucket_count - 1) + 1).round().astype(int)
    return scored


def _segment_customer(row: pd.Series) -> str:
    r_score = int(row["R"])
    f_score = int(row["F"])
    m_score = int(row["M"])

    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return "Champions"
    if r_score >= 3 and f_score >= 4:
        return "Loyal Customers"
    if r_score >= 4 and f_score >= 2:
        return "Potential Loyalists"
    if r_score <= 2 and (f_score >= 3 or m_score >= 3):
        return "At Risk"
    return "Hibernating"


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["CustomerIDStr"] != ""].copy()
    if base.empty:
        return pd.DataFrame(columns=["CustomerIDStr", "recency_days", "frequency", "monetary", "R", "F", "M", "RFMScore", "Segment"])

    reference_date = base["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = (
        base.groupby("CustomerIDStr", as_index=False)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("InvoiceNoStr", "nunique"),
            monetary=("Sales", "sum"),
        )
    )
    rfm["recency_days"] = (reference_date - rfm["last_purchase"].dt.normalize()).dt.days
    rfm["R"] = _score_series(rfm["recency_days"], ascending=False)
    rfm["F"] = _score_series(rfm["frequency"], ascending=True)
    rfm["M"] = _score_series(rfm["monetary"], ascending=True)
    rfm["RFMScore"] = rfm[["R", "F", "M"]].astype(str).agg("".join, axis=1)
    rfm["Segment"] = rfm.apply(_segment_customer, axis=1)
    return rfm.sort_values(["monetary", "frequency"], ascending=False)


def summarize_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    if rfm.empty:
        return pd.DataFrame(columns=["Segment", "customers", "sales", "share"])
    summary = (
        rfm.groupby("Segment", as_index=False)
        .agg(customers=("CustomerIDStr", "nunique"), sales=("monetary", "sum"))
        .sort_values("sales", ascending=False)
    )
    total_sales = float(summary["sales"].sum())
    summary["share"] = summary["sales"] / total_sales if total_sales else 0.0
    return summary

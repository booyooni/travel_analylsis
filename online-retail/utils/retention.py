from __future__ import annotations

import pandas as pd


def build_retention_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[df["CustomerIDStr"] != ""].copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    base["InvoiceMonth"] = base["InvoiceDate"].dt.to_period("M")
    first_month = base.groupby("CustomerIDStr")["InvoiceMonth"].min().rename("CohortMonth")
    base = base.merge(first_month, on="CustomerIDStr", how="left")
    base["CohortIndex"] = (base["InvoiceMonth"] - base["CohortMonth"]).apply(lambda x: x.n)

    cohort_counts = (
        base.groupby(["CohortMonth", "CohortIndex"])["CustomerIDStr"]
        .nunique()
        .reset_index()
        .pivot(index="CohortMonth", columns="CohortIndex", values="CustomerIDStr")
        .sort_index()
        .fillna(0)
    )
    if 0 not in cohort_counts.columns:
        cohort_counts[0] = 0
        cohort_counts = cohort_counts.sort_index(axis=1)
    cohort_sizes = cohort_counts[0].replace(0, pd.NA)
    retention = cohort_counts.divide(cohort_sizes, axis=0).fillna(0)

    cohort_counts.index = cohort_counts.index.astype(str)
    retention.index = retention.index.astype(str)
    return cohort_counts, retention

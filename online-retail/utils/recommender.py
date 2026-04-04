from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str).str.strip().str.lower()


def build_content_catalog(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["DescriptionStr"] != "Unknown"].copy()
    if base.empty:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "sales", "quantity", "orders", "customers", "feature_text"])

    catalog = (
        base.groupby("StockCodeStr", as_index=False)
        .agg(
            DescriptionStr=("DescriptionStr", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            sales=("Sales", "sum"),
            quantity=("Quantity", "sum"),
            orders=("InvoiceNoStr", "nunique"),
            customers=("CustomerIDStr", lambda s: s[s != ""].nunique()),
        )
        .sort_values(["sales", "orders", "quantity"], ascending=False)
        .reset_index(drop=True)
    )
    normalized_name = _normalize_text(catalog["DescriptionStr"])
    catalog["feature_text"] = normalized_name + " " + catalog["StockCodeStr"].astype(str).str.lower()
    return catalog


def build_content_model(catalog: pd.DataFrame) -> dict[str, object]:
    if catalog.empty:
        return {"catalog": catalog, "vectorizer": None, "tfidf": None, "index_by_code": {}}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, norm="l2")
    tfidf = vectorizer.fit_transform(catalog["feature_text"])
    index_by_code = {code: idx for idx, code in enumerate(catalog["StockCodeStr"].tolist())}
    return {
        "catalog": catalog,
        "vectorizer": vectorizer,
        "tfidf": tfidf,
        "index_by_code": index_by_code,
    }


def recommend_similar_products(content_model: dict[str, object], stock_code: str, top_n: int = 10) -> pd.DataFrame:
    catalog = content_model.get("catalog")
    tfidf = content_model.get("tfidf")
    index_by_code = content_model.get("index_by_code")

    if not isinstance(catalog, pd.DataFrame) or catalog.empty or tfidf is None or not isinstance(index_by_code, dict):
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "similarity", "sales", "orders", "quantity"])

    if stock_code not in index_by_code:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "similarity", "sales", "orders", "quantity"])

    src_idx = index_by_code[stock_code]
    sims = linear_kernel(tfidf[src_idx], tfidf).ravel()
    similar_idx = np.argsort(-sims)

    rows: list[dict[str, object]] = []
    for idx in similar_idx:
        if idx == src_idx:
            continue
        row = catalog.iloc[int(idx)]
        rows.append(
            {
                "StockCodeStr": row["StockCodeStr"],
                "DescriptionStr": row["DescriptionStr"],
                "similarity": float(sims[int(idx)]),
                "sales": float(row["sales"]),
                "orders": int(row["orders"]),
                "quantity": float(row["quantity"]),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def build_cf_model(df: pd.DataFrame) -> dict[str, object]:
    base = df[(df["CustomerIDStr"] != "") & (df["DescriptionStr"] != "Unknown")].copy()
    if base.empty:
        return {
            "matrix": csr_matrix((0, 0)),
            "customer_ids": [],
            "stock_codes": [],
            "stock_to_idx": {},
            "knn": None,
            "agg": pd.DataFrame(),
        }

    agg = (
        base.groupby(["CustomerIDStr", "StockCodeStr"], as_index=False)
        .agg(
            quantity=("Quantity", "sum"),
            sales=("Sales", "sum"),
            DescriptionStr=("DescriptionStr", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
    )

    customer_ids = sorted(agg["CustomerIDStr"].unique().tolist())
    stock_codes = sorted(agg["StockCodeStr"].unique().tolist())
    customer_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    stock_to_idx = {code: i for i, code in enumerate(stock_codes)}

    rows = agg["CustomerIDStr"].map(customer_to_idx).astype(int).to_numpy()
    cols = agg["StockCodeStr"].map(stock_to_idx).astype(int).to_numpy()
    # Purchase signal combines quantity and order value while dampening outliers.
    values = np.log1p(agg["quantity"].clip(lower=0).to_numpy()) + np.log1p(agg["sales"].clip(lower=0).to_numpy())

    matrix = csr_matrix((values, (rows, cols)), shape=(len(customer_ids), len(stock_codes)))

    knn = None
    if matrix.shape[0] > 1 and matrix.shape[1] > 1:
        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(matrix.T)

    return {
        "matrix": matrix,
        "customer_ids": customer_ids,
        "stock_codes": stock_codes,
        "stock_to_idx": stock_to_idx,
        "knn": knn,
        "agg": agg,
    }


def purchased_products_for_customer(df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    base = df[(df["CustomerIDStr"] == customer_id) & (df["DescriptionStr"] != "Unknown")].copy()
    if base.empty:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "orders", "quantity", "sales", "last_purchase"])

    return (
        base.groupby("StockCodeStr", as_index=False)
        .agg(
            DescriptionStr=("DescriptionStr", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            orders=("InvoiceNoStr", "nunique"),
            quantity=("Quantity", "sum"),
            sales=("Sales", "sum"),
            last_purchase=("InvoiceDate", "max"),
        )
        .sort_values(["sales", "orders", "quantity"], ascending=False)
    )


def recommend_products_cf(cf_model: dict[str, object], customer_id: str, top_n: int = 10, item_neighbors: int = 15) -> pd.DataFrame:
    matrix = cf_model.get("matrix")
    customer_ids = cf_model.get("customer_ids")
    stock_codes = cf_model.get("stock_codes")
    stock_to_idx = cf_model.get("stock_to_idx")
    knn = cf_model.get("knn")
    agg = cf_model.get("agg")

    if (
        not isinstance(matrix, csr_matrix)
        or not isinstance(customer_ids, list)
        or not isinstance(stock_codes, list)
        or not isinstance(stock_to_idx, dict)
        or not isinstance(agg, pd.DataFrame)
    ):
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "score"]) 

    if customer_id not in customer_ids or matrix.shape[1] == 0 or knn is None:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "score"]) 

    customer_idx = customer_ids.index(customer_id)
    user_vector = matrix.getrow(customer_idx)
    purchased_idx = set(user_vector.indices.tolist())
    if not purchased_idx:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "score"]) 

    score_map: dict[int, float] = {}
    for item_idx in purchased_idx:
        n_neighbors = min(item_neighbors + 1, matrix.shape[1])
        distances, indices = knn.kneighbors(matrix.T.getrow(item_idx), n_neighbors=n_neighbors)
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if int(neighbor_idx) in purchased_idx:
                continue
            sim = 1.0 - float(dist)
            if sim <= 0:
                continue
            score_map[int(neighbor_idx)] = score_map.get(int(neighbor_idx), 0.0) + sim

    if not score_map:
        return pd.DataFrame(columns=["StockCodeStr", "DescriptionStr", "score"]) 

    top_items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    desc_map = (
        agg.groupby("StockCodeStr")["DescriptionStr"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )

    rows = []
    for item_idx, score in top_items:
        code = stock_codes[item_idx]
        rows.append(
            {
                "StockCodeStr": code,
                "DescriptionStr": desc_map.get(code, "Unknown"),
                "score": float(score),
            }
        )

    return pd.DataFrame(rows)

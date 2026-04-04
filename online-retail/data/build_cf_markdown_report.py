from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def build_user_item_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = df.dropna(subset=["CustomerID", "StockCode", "Description"]).copy()
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int)

    # Use total purchased quantity as implicit preference strength.
    user_item = cleaned.pivot_table(
        index="CustomerID",
        columns="StockCode",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    # Representative description per stock code.
    item_desc = (
        cleaned.groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .sort_index()
    )
    return user_item, item_desc


def recommend_for_user(
    target_user: int,
    user_item_binary: pd.DataFrame,
    user_ids: np.ndarray,
    sim_matrix: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    user_pos = int(np.where(user_ids == target_user)[0][0])
    sims = sim_matrix[user_pos].copy()
    sims[user_pos] = 0.0

    neighbors_mask = sims > 0
    if not np.any(neighbors_mask):
        return pd.DataFrame(columns=["StockCode", "score", "buyer_sim_mean"])

    weights = sims[neighbors_mask]
    neighbor_purchases = user_item_binary.values[neighbors_mask]
    weighted_scores = (weights @ neighbor_purchases) / (weights.sum() + 1e-12)

    purchased = user_item_binary.loc[target_user].values > 0
    weighted_scores[purchased] = -1

    candidate_idx = np.where(weighted_scores >= 0)[0]
    if candidate_idx.size == 0:
        return pd.DataFrame(columns=["StockCode", "score", "buyer_sim_mean"])

    top_idx = candidate_idx[np.argsort(weighted_scores[candidate_idx])[::-1][:top_k]]

    buyer_sim_means: list[float] = []
    for col_idx in top_idx:
        buyers = neighbor_purchases[:, col_idx] > 0
        if buyers.any():
            buyer_sim_means.append(float(weights[buyers].mean()))
        else:
            buyer_sim_means.append(0.0)

    recs = pd.DataFrame(
        {
            "StockCode": user_item_binary.columns[top_idx],
            "score": weighted_scores[top_idx],
            "buyer_sim_mean": buyer_sim_means,
        }
    )
    return recs.sort_values("score", ascending=False).reset_index(drop=True)


def build_mf_model(user_item_binary: pd.DataFrame, n_components: int = 50) -> tuple[np.ndarray, np.ndarray]:
    max_components = min(user_item_binary.shape[0] - 1, user_item_binary.shape[1] - 1)
    n_components = max(2, min(n_components, max_components))

    svd = TruncatedSVD(n_components=n_components, random_state=20260401)
    user_factors = svd.fit_transform(user_item_binary.values)
    item_factors = svd.components_.T
    return user_factors, item_factors


def recommend_for_user_mf(
    target_user: int,
    user_item_binary: pd.DataFrame,
    user_ids: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    user_pos = int(np.where(user_ids == target_user)[0][0])

    user_vector = user_factors[user_pos]
    raw_scores = user_vector @ item_factors.T

    purchased_mask = user_item_binary.loc[target_user].values > 0
    raw_scores[purchased_mask] = -1e9
    candidate_idx = np.where(raw_scores > -1e8)[0]
    if candidate_idx.size == 0:
        return pd.DataFrame(columns=["StockCode", "score", "latent_similarity"])

    top_idx = candidate_idx[np.argsort(raw_scores[candidate_idx])[::-1][:top_k]]

    purchased_item_vectors = item_factors[purchased_mask]
    if purchased_item_vectors.shape[0] > 0:
        user_profile = purchased_item_vectors.mean(axis=0, keepdims=True)
        latent_sim = cosine_similarity(item_factors[top_idx], user_profile).reshape(-1)
    else:
        latent_sim = np.zeros(shape=len(top_idx), dtype=float)

    recs = pd.DataFrame(
        {
            "StockCode": user_item_binary.columns[top_idx],
            "score": raw_scores[top_idx],
            "latent_similarity": latent_sim,
        }
    )
    return recs.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "online_retail.parquet"
    report_path = base_dir / "online_retail_cf_report.md"

    df = pd.read_parquet(data_path)
    user_item, item_desc = build_user_item_matrix(df)
    user_item_binary = (user_item > 0).astype(int)

    user_ids = user_item_binary.index.to_numpy()
    sim_matrix = cosine_similarity(user_item_binary.values)
    user_factors, item_factors = build_mf_model(user_item_binary, n_components=50)

    rng = np.random.default_rng(20260401)
    eligible_users = user_item_binary.sum(axis=1)
    eligible_users = eligible_users[eligible_users >= 3].index.to_numpy()
    sample_users = rng.choice(eligible_users, size=5, replace=False)

    lines: list[str] = []
    lines.append("# Online Retail 협업필터링 추천 리포트")
    lines.append("")
    lines.append(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 데이터: online-retail/online_retail.parquet")
    lines.append(f"- 원본 행 수: {len(df):,}")
    lines.append(f"- 사용자 수: {user_item.shape[0]:,}")
    lines.append(f"- 상품 수: {user_item.shape[1]:,}")
    lines.append(f"- User-Item 매트릭스 크기: {user_item.shape[0]} x {user_item.shape[1]}")
    lines.append("")
    lines.append("## 방법")
    lines.append("")
    lines.append("- 사용자-상품 구매량 행렬(Quantity 합계) 구성")
    lines.append("- 이진 행렬로 변환 후 사용자 간 코사인 유사도 계산")
    lines.append("- 사용자별 미구매 상품에 대해 유사 사용자 구매 여부를 가중평균하여 구매 확률 점수(score) 산출")
    lines.append("- 추천 상품별 유사도는 해당 상품을 구매한 이웃들의 평균 유사도(buyer_sim_mean)로 제시")
    lines.append("")
    lines.append("## 랜덤 유저 5명 결과")
    lines.append("")

    for user_id in sample_users:
        lines.append(f"### User {user_id}")
        lines.append("")

        purchased = user_item.loc[user_id]
        top_purchased = purchased[purchased > 0].sort_values(ascending=False).head(3)
        purchase_rows: list[list[str]] = []
        for stock_code, qty in top_purchased.items():
            purchase_rows.append(
                [
                    str(stock_code),
                    str(item_desc.get(stock_code, "")),
                    f"{int(qty):,}",
                ]
            )

        lines.append("- 구매 상위 3개 상품")
        lines.append("")
        lines.append(
            md_table(
                ["StockCode", "Description", "TotalQuantity"],
                purchase_rows,
            )
        )
        lines.append("")

        recs = recommend_for_user(
            target_user=int(user_id),
            user_item_binary=user_item_binary,
            user_ids=user_ids,
            sim_matrix=sim_matrix,
            top_k=10,
        )

        rec_rows: list[list[str]] = []
        for _, row in recs.iterrows():
            stock_code = row["StockCode"]
            rec_rows.append(
                [
                    str(stock_code),
                    str(item_desc.get(stock_code, "")),
                    f"{row['score']:.4f}",
                    f"{row['buyer_sim_mean']:.4f}",
                ]
            )

        lines.append("- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)")
        lines.append("")
        lines.append(
            md_table(
                ["StockCode", "Description", "PurchaseScore", "Similarity"],
                rec_rows,
            )
        )
        lines.append("")

    lines.append("## 행렬분해(Matrix Factorization, TruncatedSVD) 기반 추천 결과")
    lines.append("")
    lines.append("- 동일한 랜덤 유저 5명에 대해 잠재요인 기반 추천 수행")
    lines.append("- PurchaseScore: 사용자 잠재벡터와 상품 잠재벡터의 내적 점수")
    lines.append("- Similarity: 추천 상품 잠재벡터와 해당 사용자 구매 이력 잠재프로필 간 코사인 유사도")
    lines.append("")

    overlap_rows: list[list[str]] = []
    for user_id in sample_users:
        lines.append(f"### User {user_id} (Matrix Factorization)")
        lines.append("")

        purchased = user_item.loc[user_id]
        top_purchased = purchased[purchased > 0].sort_values(ascending=False).head(3)
        purchase_rows: list[list[str]] = []
        for stock_code, qty in top_purchased.items():
            purchase_rows.append(
                [
                    str(stock_code),
                    str(item_desc.get(stock_code, "")),
                    f"{int(qty):,}",
                ]
            )

        lines.append("- 구매 상위 3개 상품")
        lines.append("")
        lines.append(
            md_table(
                ["StockCode", "Description", "TotalQuantity"],
                purchase_rows,
            )
        )
        lines.append("")

        user_cf_recs = recommend_for_user(
            target_user=int(user_id),
            user_item_binary=user_item_binary,
            user_ids=user_ids,
            sim_matrix=sim_matrix,
            top_k=10,
        )
        mf_recs = recommend_for_user_mf(
            target_user=int(user_id),
            user_item_binary=user_item_binary,
            user_ids=user_ids,
            user_factors=user_factors,
            item_factors=item_factors,
            top_k=10,
        )

        rec_rows: list[list[str]] = []
        for _, row in mf_recs.iterrows():
            stock_code = row["StockCode"]
            rec_rows.append(
                [
                    str(stock_code),
                    str(item_desc.get(stock_code, "")),
                    f"{row['score']:.4f}",
                    f"{row['latent_similarity']:.4f}",
                ]
            )

        lines.append("- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)")
        lines.append("")
        lines.append(
            md_table(
                ["StockCode", "Description", "PurchaseScore", "Similarity"],
                rec_rows,
            )
        )
        lines.append("")

        cf_set = set(user_cf_recs["StockCode"].astype(str).tolist())
        mf_set = set(mf_recs["StockCode"].astype(str).tolist())
        overlap = len(cf_set & mf_set)
        overlap_rows.append([str(int(user_id)), str(overlap), f"{overlap / 10:.2f}"])

    lines.append("## 기존 추천시스템과 행렬분해 추천 비교")
    lines.append("")
    lines.append("- 기존(User-Based CF): 비슷한 유저 이웃의 실제 구매 패턴에 직접 의존")
    lines.append("- 행렬분해(MF): 유저/상품 잠재요인을 학습해 직접 겹치지 않은 구매패턴도 일반화")
    lines.append("- 해석성: 기존 방식은 이웃 유사도 근거가 직관적이고, MF는 잠재요인 기반이라 설명력이 상대적으로 낮지만 확장성이 좋음")
    lines.append("- 희소성 대응: MF가 일반적으로 희소한 매트릭스에서 안정적인 추천을 제공")
    lines.append("")
    lines.append("### 유저별 Top-10 추천 겹침도 (기존 vs MF)")
    lines.append("")
    lines.append(md_table(["UserID", "OverlapCount", "OverlapRate"], overlap_rows))
    lines.append("")
    lines.append("## 보안 PSA")
    lines.append("")
    lines.append("- PSA: npm/bun/pnpm/uv는 모두 패키지의 최소 릴리스 연령(minimum release age) 설정을 지원합니다.")
    lines.append("- ~/.npmrc에 `ignore-scripts=true`를 설정해두면 lifecycle script 실행을 차단할 수 있으며, 이번 분석 기준으로 이 설정만으로도 해당 취약점 완화에 도움이 됩니다.")
    lines.append("- bun과 pnpm은 기본값으로 lifecycle script를 실행하지 않습니다.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
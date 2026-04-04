from __future__ import annotations

import pandas as pd


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_currency(value: int | float) -> str:
    return f"{float(value):,.2f}"


def fmt_pct(value: int | float) -> str:
    return f"{float(value):.1%}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def build_data_basis(row_count: int, columns: str, grouping: str, transforms: str) -> str:
    return (
        f"이 시각화는 현재 필터가 적용된 거래 데이터 {row_count:,}행을 기준으로 {columns} 컬럼을 사용해 "
        f"{grouping} 방식으로 집계했으며, {transforms} 과정을 거쳐 계산하였다."
    )


def build_interpretation(metric_focus: str, caution: str) -> str:
    return (
        f"이 그래프는 {metric_focus}의 높고 낮음을 비교해 패턴을 읽는 용도이며, "
        f"{caution} 점을 함께 확인하면서 해석해야 한다."
    )

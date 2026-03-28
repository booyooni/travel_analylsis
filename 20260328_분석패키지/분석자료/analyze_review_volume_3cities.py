import pandas as pd
from pathlib import Path

base = Path('.')
files = {
    '다낭': base / 'verygoodtour_reviews_danang_1year.csv',
    '나트랑': base / 'verygoodtour_reviews_nhatrang_1year.csv',
    '싱가폴': base / 'verygoodtour_reviews_singapore_1year.csv',
}

frames = []
for city, path in files.items():
    df = pd.read_csv(path)
    df['city'] = city
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df = all_df.dropna(subset=['date']).copy()
all_df['year_month'] = all_df['date'].dt.to_period('M').astype(str)

city_counts = all_df.groupby('city', as_index=False).size().rename(columns={'size': 'review_count'})
total = int(city_counts['review_count'].sum())
city_counts['share_pct'] = (city_counts['review_count'] / total * 100).round(2)
city_counts['avg_per_month'] = (city_counts['review_count'] / 12).round(2)

monthly = all_df.groupby(['year_month', 'city'], as_index=False).size().rename(columns={'size': 'review_count'})
monthly_pivot = monthly.pivot(index='year_month', columns='city', values='review_count').fillna(0).astype(int).reset_index()
for city in ['다낭', '나트랑', '싱가폴']:
    if city not in monthly_pivot.columns:
        monthly_pivot[city] = 0
monthly_pivot['합계'] = monthly_pivot[['다낭', '나트랑', '싱가폴']].sum(axis=1)

recent_3m = all_df[all_df['date'] >= (all_df['date'].max() - pd.DateOffset(months=3))]
recent_counts = recent_3m.groupby('city', as_index=False).size().rename(columns={'size': 'review_count_recent_3m'})

peak_rows = []
for city in ['다낭', '나트랑', '싱가폴']:
    sub = monthly[monthly['city'] == city]
    if sub.empty:
        continue
    peak = sub.loc[sub['review_count'].idxmax()]
    low = sub.loc[sub['review_count'].idxmin()]
    peak_rows.append(
        {
            'city': city,
            'peak_month': peak['year_month'],
            'peak_count': int(peak['review_count']),
            'low_month': low['year_month'],
            'low_count': int(low['review_count']),
        }
    )
peak_df = pd.DataFrame(peak_rows)

city_summary = city_counts.merge(recent_counts, on='city', how='left').fillna({'review_count_recent_3m': 0})
city_summary = city_summary.merge(peak_df, on='city', how='left')
city_summary['review_count_recent_3m'] = city_summary['review_count_recent_3m'].astype(int)

city_summary.to_csv('reviews_volume_summary_3cities.csv', index=False, encoding='utf-8-sig')
monthly_pivot.to_csv('reviews_volume_monthly_3cities.csv', index=False, encoding='utf-8-sig')

lines = []
lines.append('# 다낭/나트랑/싱가폴 리뷰량 분석')
lines.append('')
lines.append(f'- 분석 기간: {all_df["date"].min().date()} ~ {all_df["date"].max().date()}')
lines.append(f'- 총 리뷰 수: {total:,}건')
lines.append('')
lines.append('## 1) 도시별 리뷰량')
lines.append('')
lines.append('| 도시 | 리뷰 수 | 점유율(%) | 월평균(건) | 최근 3개월(건) |')
lines.append('|---|---:|---:|---:|---:|')
for _, row in city_summary.sort_values('review_count', ascending=False).iterrows():
    lines.append(
        f"| {row['city']} | {int(row['review_count']):,} | {row['share_pct']:.2f} | {row['avg_per_month']:.2f} | {int(row['review_count_recent_3m']):,} |"
    )

lines.append('')
lines.append('## 2) 도시별 피크/저점 월')
lines.append('')
lines.append('| 도시 | 피크 월 | 피크 건수 | 저점 월 | 저점 건수 |')
lines.append('|---|---|---:|---|---:|')
for _, row in peak_df.iterrows():
    lines.append(
        f"| {row['city']} | {row['peak_month']} | {int(row['peak_count'])} | {row['low_month']} | {int(row['low_count'])} |"
    )

lines.append('')
lines.append('## 3) 핵심 인사이트')
lines.append('')
ranked = city_summary.sort_values('review_count', ascending=False)['city'].tolist()
lines.append(f"- 리뷰량 순위: {' > '.join(ranked)}")
first = city_summary.sort_values('review_count', ascending=False).iloc[0]
last = city_summary.sort_values('review_count', ascending=True).iloc[0]
multiple = first['review_count'] / last['review_count'] if last['review_count'] else 0
lines.append(f"- 최대/최소 도시 격차: 약 {multiple:.2f}배")
lines.append('- 최근 3개월 건수는 최근 관심도 신호로 활용 가능')

Path('리뷰양_분석_다낭_나트랑_싱가폴.md').write_text('\n'.join(lines), encoding='utf-8')

print('created: reviews_volume_summary_3cities.csv')
print('created: reviews_volume_monthly_3cities.csv')
print('created: 리뷰양_분석_다낭_나트랑_싱가폴.md')
print('\n도시별 요약')
print(city_summary.sort_values('review_count', ascending=False).to_string(index=False))

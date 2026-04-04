# Online Retail 협업필터링 추천 리포트

- 생성 시각: 2026-04-01 21:00:16
- 데이터: online-retail/online_retail.parquet
- 원본 행 수: 541,909
- 사용자 수: 4,338
- 상품 수: 3,665
- User-Item 매트릭스 크기: 4338 x 3665

## 방법

- 사용자-상품 구매량 행렬(Quantity 합계) 구성
- 이진 행렬로 변환 후 사용자 간 코사인 유사도 계산
- 사용자별 미구매 상품에 대해 유사 사용자 구매 여부를 가중평균하여 구매 확률 점수(score) 산출
- 추천 상품별 유사도는 해당 상품을 구매한 이웃들의 평균 유사도(buyer_sim_mean)로 제시

## 랜덤 유저 5명 결과

### User 16900

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 16045 | POPART WOODEN PENCILS ASST | 100 |
| 20868 | SILVER FABRIC MIRROR | 60 |
| 51014A | FEATHER PEN,HOT PINK | 36 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 22423 | REGENCY CAKESTAND 3 TIER | 0.3018 | 0.0638 |
| 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 0.2578 | 0.0595 |
| 21212 | PACK OF 72 RETROSPOT CAKE CASES | 0.2507 | 0.0680 |
| 23298 | SPOTTY BUNTING | 0.2131 | 0.0664 |
| 85099B | JUMBO BAG RED RETROSPOT | 0.2119 | 0.0630 |
| 22720 | SET OF 3 CAKE TINS PANTRY DESIGN  | 0.2105 | 0.0619 |
| 22382 | LUNCH BAG SPACEBOY DESIGN  | 0.1950 | 0.0687 |
| 22086 | PAPER CHAIN KIT 50'S CHRISTMAS  | 0.1945 | 0.0602 |
| 22469 | HEART OF WICKER SMALL | 0.1929 | 0.0635 |
| 84879 | ASSORTED COLOUR BIRD ORNAMENT | 0.1895 | 0.0562 |

### User 16204

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 23078 | ICE CREAM PEN LIP GLOSS  | 24 |
| 23501 | KEY RING BASEBALL BOOT UNION JACK | 20 |
| 20676 | RED RETROSPOT BOWL | 6 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 0.2619 | 0.0499 |
| 22423 | REGENCY CAKESTAND 3 TIER | 0.2609 | 0.0475 |
| 21212 | PACK OF 72 RETROSPOT CAKE CASES | 0.2553 | 0.0528 |
| 85099B | JUMBO BAG RED RETROSPOT | 0.2216 | 0.0510 |
| 47566 | PARTY BUNTING | 0.2188 | 0.0503 |
| 23355 | HOT WATER BOTTLE KEEP CALM | 0.2157 | 0.0577 |
| 22086 | PAPER CHAIN KIT 50'S CHRISTMAS  | 0.2150 | 0.0513 |
| 20725 | LUNCH BAG RED RETROSPOT | 0.2084 | 0.0521 |
| 22720 | SET OF 3 CAKE TINS PANTRY DESIGN  | 0.2043 | 0.0479 |
| 23298 | SPOTTY BUNTING | 0.1940 | 0.0492 |

### User 17830

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 22464 | HANGING METAL HEART LANTERN | 12 |
| 22969 | HOMEMADE JAM SCENTED CANDLES | 12 |
| 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 12 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 22423 | REGENCY CAKESTAND 3 TIER | 0.2753 | 0.0606 |
| 47566 | PARTY BUNTING | 0.2349 | 0.0599 |
| 22469 | HEART OF WICKER SMALL | 0.2286 | 0.0661 |
| 22720 | SET OF 3 CAKE TINS PANTRY DESIGN  | 0.2196 | 0.0610 |
| 84879 | ASSORTED COLOUR BIRD ORNAMENT | 0.2125 | 0.0595 |
| 22470 | HEART OF WICKER LARGE | 0.2038 | 0.0661 |
| 85099B | JUMBO BAG RED RETROSPOT | 0.1962 | 0.0591 |
| 21733 | RED HANGING HEART T-LIGHT HOLDER | 0.1810 | 0.0656 |
| 23298 | SPOTTY BUNTING | 0.1803 | 0.0560 |
| 82482 | WOODEN PICTURE FRAME WHITE FINISH | 0.1796 | 0.0677 |

### User 15204

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 22544 | MINI JIGSAW SPACEBOY | 96 |
| 22029 | SPACEBOY BIRTHDAY CARD | 48 |
| 21062 | PARTY INVITES SPACEMAN | 36 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 22138 | BAKING SET 9 PIECE RETROSPOT  | 0.3743 | 0.0709 |
| 20725 | LUNCH BAG RED RETROSPOT | 0.3172 | 0.0593 |
| 22629 | SPACEBOY LUNCH BOX  | 0.2809 | 0.0745 |
| 20727 | LUNCH BAG  BLACK SKULL. | 0.2722 | 0.0586 |
| 20728 | LUNCH BAG CARS BLUE | 0.2711 | 0.0559 |
| 22383 | LUNCH BAG SUKI DESIGN  | 0.2675 | 0.0574 |
| 20726 | LUNCH BAG WOODLAND | 0.2642 | 0.0632 |
| 22423 | REGENCY CAKESTAND 3 TIER | 0.2548 | 0.0575 |
| 21212 | PACK OF 72 RETROSPOT CAKE CASES | 0.2494 | 0.0537 |
| 22551 | PLASTERS IN TIN SPACEBOY | 0.2493 | 0.0693 |

### User 15213

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 16161P | WRAP ENGLISH ROSE  | 25 |
| 21498 | RED RETROSPOT WRAP  | 25 |
| 84991 | 60 TEATIME FAIRY CAKE CASES | 24 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 20728 | LUNCH BAG CARS BLUE | 0.2791 | 0.0959 |
| 22384 | LUNCH BAG PINK POLKADOT | 0.2691 | 0.0993 |
| 22383 | LUNCH BAG SUKI DESIGN  | 0.2633 | 0.0993 |
| 20727 | LUNCH BAG  BLACK SKULL. | 0.2630 | 0.0978 |
| 23203 | JUMBO BAG VINTAGE DOILY  | 0.2602 | 0.0912 |
| 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 0.2593 | 0.0700 |
| 22423 | REGENCY CAKESTAND 3 TIER | 0.2579 | 0.0693 |
| 23209 | LUNCH BAG VINTAGE DOILY  | 0.2507 | 0.0925 |
| 22720 | SET OF 3 CAKE TINS PANTRY DESIGN  | 0.2420 | 0.0741 |
| 47566 | PARTY BUNTING | 0.2375 | 0.0746 |

## 행렬분해(Matrix Factorization, TruncatedSVD) 기반 추천 결과

- 동일한 랜덤 유저 5명에 대해 잠재요인 기반 추천 수행
- PurchaseScore: 사용자 잠재벡터와 상품 잠재벡터의 내적 점수
- Similarity: 추천 상품 잠재벡터와 해당 사용자 구매 이력 잠재프로필 간 코사인 유사도

### User 16900 (Matrix Factorization)

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 16045 | POPART WOODEN PENCILS ASST | 100 |
| 20868 | SILVER FABRIC MIRROR | 60 |
| 51014A | FEATHER PEN,HOT PINK | 36 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 23298 | SPOTTY BUNTING | 0.4447 | 0.3064 |
| 22423 | REGENCY CAKESTAND 3 TIER | 0.3867 | 0.1903 |
| 21977 | PACK OF 60 PINK PAISLEY CAKE CASES | 0.3322 | 0.2853 |
| 21212 | PACK OF 72 RETROSPOT CAKE CASES | 0.3312 | 0.2233 |
| 23084 | RABBIT NIGHT LIGHT | 0.3271 | 0.2513 |
| 23355 | HOT WATER BOTTLE KEEP CALM | 0.3186 | 0.2402 |
| 23170 | REGENCY TEA PLATE ROSES  | 0.2999 | 0.3212 |
| 23175 | REGENCY MILK JUG PINK  | 0.2945 | 0.3358 |
| 22469 | HEART OF WICKER SMALL | 0.2901 | 0.1909 |
| 23172 | REGENCY TEA PLATE PINK | 0.2787 | 0.3438 |

### User 16204 (Matrix Factorization)

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 23078 | ICE CREAM PEN LIP GLOSS  | 24 |
| 23501 | KEY RING BASEBALL BOOT UNION JACK | 20 |
| 20676 | RED RETROSPOT BOWL | 6 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 22726 | ALARM CLOCK BAKELIKE GREEN | 0.4663 | 0.5200 |
| 22730 | ALARM CLOCK BAKELIKE IVORY | 0.3442 | 0.4962 |
| 22866 | HAND WARMER SCOTTY DOG DESIGN | 0.2919 | 0.3337 |
| 22729 | ALARM CLOCK BAKELIKE ORANGE | 0.2812 | 0.5022 |
| 22867 | HAND WARMER BIRD DESIGN | 0.2635 | 0.3079 |
| 23439 | HAND WARMER RED LOVE HEART | 0.2587 | 0.3765 |
| 22632 | HAND WARMER RED RETROSPOT | 0.2452 | 0.3693 |
| 21212 | PACK OF 72 RETROSPOT CAKE CASES | 0.1975 | 0.1999 |
| 23355 | HOT WATER BOTTLE KEEP CALM | 0.1918 | 0.2170 |
| 22834 | HAND WARMER BABUSHKA DESIGN | 0.1841 | 0.3371 |

### User 17830 (Matrix Factorization)

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 22464 | HANGING METAL HEART LANTERN | 12 |
| 22969 | HOMEMADE JAM SCENTED CANDLES | 12 |
| 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 12 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 48194 | DOORMAT HEARTS | 0.3161 | 0.6500 |
| 20685 | DOORMAT RED RETROSPOT | 0.3025 | 0.5788 |
| 21733 | RED HANGING HEART T-LIGHT HOLDER | 0.2843 | 0.4895 |
| 48138 | DOORMAT UNION FLAG | 0.2743 | 0.5815 |
| 48184 | DOORMAT ENGLISH ROSE  | 0.2173 | 0.5209 |
| 23284 | DOORMAT KEEP CALM AND COME IN | 0.1993 | 0.3820 |
| 22804 | CANDLEHOLDER PINK HANGING HEART | 0.1966 | 0.4356 |
| 48188 | DOORMAT WELCOME PUPPIES | 0.1838 | 0.5632 |
| 21755 | LOVE BUILDING BLOCK WORD | 0.1799 | 0.3793 |
| 22690 | DOORMAT HOME SWEET HOME BLUE  | 0.1663 | 0.4734 |

### User 15204 (Matrix Factorization)

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 22544 | MINI JIGSAW SPACEBOY | 96 |
| 22029 | SPACEBOY BIRTHDAY CARD | 48 |
| 21062 | PARTY INVITES SPACEMAN | 36 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 22138 | BAKING SET 9 PIECE RETROSPOT  | 0.2474 | 0.4415 |
| 22629 | SPACEBOY LUNCH BOX  | 0.2128 | 0.6323 |
| 22630 | DOLLY GIRL LUNCH BOX | 0.1802 | 0.5614 |
| 22899 | CHILDREN'S APRON DOLLY GIRL  | 0.1359 | 0.5948 |
| 22662 | LUNCH BAG DOLLY GIRL DESIGN | 0.1271 | 0.4726 |
| 23290 | SPACEBOY CHILDRENS BOWL | 0.1163 | 0.6151 |
| 22139 | RETROSPOT TEA SET CERAMIC 11 PC  | 0.1096 | 0.2572 |
| 23289 | DOLLY GIRL CHILDRENS BOWL | 0.1051 | 0.5855 |
| 23256 | CHILDRENS CUTLERY SPACEBOY  | 0.1036 | 0.3758 |
| 23254 | CHILDRENS CUTLERY DOLLY GIRL  | 0.1016 | 0.4021 |

### User 15213 (Matrix Factorization)

- 구매 상위 3개 상품

| StockCode | Description | TotalQuantity |
| --- | --- | --- |
| 16161P | WRAP ENGLISH ROSE  | 25 |
| 21498 | RED RETROSPOT WRAP  | 25 |
| 84991 | 60 TEATIME FAIRY CAKE CASES | 24 |

- 향후 구매 확률 높은 추천 상품 10개 (유사도 포함)

| StockCode | Description | PurchaseScore | Similarity |
| --- | --- | --- | --- |
| 22384 | LUNCH BAG PINK POLKADOT | 0.3109 | 0.4679 |
| 21977 | PACK OF 60 PINK PAISLEY CAKE CASES | 0.2919 | 0.4080 |
| 20727 | LUNCH BAG  BLACK SKULL. | 0.2737 | 0.4043 |
| 20728 | LUNCH BAG CARS BLUE | 0.2611 | 0.3649 |
| 22960 | JAM MAKING SET WITH JARS | 0.2542 | 0.3041 |
| 22558 | CLOTHES PEGS RETROSPOT PACK 24  | 0.2538 | 0.4307 |
| 84992 | 72 SWEETHEART FAIRY CAKE CASES | 0.2334 | 0.3936 |
| 22383 | LUNCH BAG SUKI DESIGN  | 0.2295 | 0.3377 |
| 22386 | JUMBO BAG PINK POLKADOT | 0.2189 | 0.3665 |
| 20914 | SET/5 RED RETROSPOT LID GLASS BOWLS | 0.2165 | 0.4013 |

## 기존 추천시스템과 행렬분해 추천 비교

- 기존(User-Based CF): 비슷한 유저 이웃의 실제 구매 패턴에 직접 의존
- 행렬분해(MF): 유저/상품 잠재요인을 학습해 직접 겹치지 않은 구매패턴도 일반화
- 해석성: 기존 방식은 이웃 유사도 근거가 직관적이고, MF는 잠재요인 기반이라 설명력이 상대적으로 낮지만 확장성이 좋음
- 희소성 대응: MF가 일반적으로 희소한 매트릭스에서 안정적인 추천을 제공

### 유저별 Top-10 추천 겹침도 (기존 vs MF)

| UserID | OverlapCount | OverlapRate |
| --- | --- | --- |
| 16900 | 4 | 0.40 |
| 16204 | 2 | 0.20 |
| 17830 | 1 | 0.10 |
| 15204 | 2 | 0.20 |
| 15213 | 4 | 0.40 |

## 보안 PSA

- PSA: npm/bun/pnpm/uv는 모두 패키지의 최소 릴리스 연령(minimum release age) 설정을 지원합니다.
- ~/.npmrc에 `ignore-scripts=true`를 설정해두면 lifecycle script 실행을 차단할 수 있으며, 이번 분석 기준으로 이 설정만으로도 해당 취약점 완화에 도움이 됩니다.
- bun과 pnpm은 기본값으로 lifecycle script를 실행하지 않습니다.

import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('label_eda_full.csv')
print(f"총 레코드: {len(df):,}")
print(f"원본: {(~df['is_aug']).sum():,}")
print(f"증강: {df['is_aug'].sum():,}")
print()

# 1. crop_folder → crop_code 매핑
print("=== crop_folder -> crop_code 매핑 ===")
mapping = df.groupby(['env','crop_folder','crop_code']).size().reset_index(name='count')
print(mapping.to_string(index=False))
print()

# 2. 원본 기준 작물×환경별 risk 분포 (train split)
orig_train = df[(~df['is_aug']) & (df['split']=='train')]
pivot = (
    orig_train.groupby(['env','crop_folder','risk'])
    .size().unstack(fill_value=0)
)
for c in [0,1,2,3]:
    if c not in pivot.columns:
        pivot[c] = 0
pivot = pivot[[0,1,2,3]]
pivot.columns = ['정상(0)','초기(1)','중기(2)','말기(3)']
pivot['min_class'] = pivot.min(axis=1)
pivot['total'] = pivot.sum(axis=1)
pivot = pivot.reset_index().sort_values('min_class', ascending=False)
print("=== 원본 Train 기준 작물×환경별 risk 분포 (min_class 내림차순) ===")
print(pivot.to_string(index=False))
print()

# 3. 증강 포함 전체 기준 risk 분포
all_train = df[df['split']=='train']
pivot2 = (
    all_train.groupby(['env','crop_folder','risk'])
    .size().unstack(fill_value=0)
)
for c in [0,1,2,3]:
    if c not in pivot2.columns:
        pivot2[c] = 0
pivot2 = pivot2[[0,1,2,3]]
pivot2.columns = ['정상(0)','초기(1)','중기(2)','말기(3)']
pivot2['min_class'] = pivot2.min(axis=1)
pivot2['total'] = pivot2.sum(axis=1)
pivot2 = pivot2.reset_index().sort_values('min_class', ascending=False)
print("=== 증강 포함 Train 기준 작물×환경별 risk 분포 (min_class 내림차순) ===")
print(pivot2.to_string(index=False))
print()

# 4. disease 분포 (원본 train 기준)
print("=== disease별 risk 분포 (원본 train) ===")
dp = (
    orig_train.groupby(['env','crop_folder','disease','risk'])
    .size().unstack(fill_value=0)
)
for c in [0,1,2,3]:
    if c not in dp.columns:
        dp[c] = 0
dp = dp[[0,1,2,3]]
dp.columns = ['정상(0)','초기(1)','중기(2)','말기(3)']
dp['min_class'] = dp.min(axis=1)
dp['total'] = dp.sum(axis=1)
dp = dp.reset_index().sort_values(['env','crop_folder','disease'])
print(dp.to_string(index=False))

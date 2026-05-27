"""
D-1 옵션 종합 분석
  - 원본 ID 기준 70:15:15 분할
  - 같은 원본의 증강은 같은 split에 배치
  - MAX_PER_CLASS=4000 캡은 train에만 적용
"""

import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

MAX_PER_CLASS = 4000

df = pd.read_csv('label_eda_full.csv')

# ─────────────────────────────────────────
# 그룹 정의
# ─────────────────────────────────────────
TRAIN_GROUPS = [
    ('시설','05.상추'), ('시설','02.고추'), ('시설','09.쥬키니호박'),
    ('시설','04.딸기'), ('시설','11.토마토'), ('시설','03.단호박'),
    ('노지','09.파'), ('노지','03.배추'), ('노지','01.고추'),
    ('노지','04.애호박'), ('노지','06.오이'), ('노지','08.콩'),
]

HELDOUT_GROUPS = [
    ('시설','01.가지'), ('시설','06.수박'), ('시설','10.참외'),
    ('노지','05.양배추'), ('노지','07.잎마름병(토마토)'), ('노지','10.호박'),
]


# ─────────────────────────────────────────
# 1. 학습 12그룹 D-1 분할 + 캡 적용 시뮬레이션
# ─────────────────────────────────────────
print("="*110)
print("【1】 학습 12그룹 D-1 분할 + MAX_PER_CLASS=4000 캡 적용")
print("="*110)

train_df = df[df['split']=='train']
results = []

for env, crop in TRAIN_GROUPS:
    g = train_df[(train_df['env']==env) & (train_df['crop_folder']==crop)]
    for risk in [0,1,2,3]:
        c = g[g['risk']==risk]
        orig = (~c['is_aug']).sum()
        aug = c['is_aug'].sum()
        total = orig + aug

        # D-1: 원본 70:15:15, 증강 따라감
        train_o = int(orig * 0.70)
        val_o   = int(orig * 0.15)
        test_o  = orig - train_o - val_o
        if orig > 0:
            train_a = int(aug * train_o / orig)
            val_a   = int(aug * val_o / orig)
            test_a  = aug - train_a - val_a
        else:
            train_a = val_a = test_a = 0

        train_raw = train_o + train_a
        val_total = val_o + val_a
        test_total = test_o + test_a

        # 캡 적용 (train만)
        train_capped = min(train_raw, MAX_PER_CLASS)

        results.append({
            'env': env, 'crop': crop, 'risk': risk,
            '원본': orig, '증강': aug,
            'train_원본': train_raw,
            'train_캡후': train_capped,
            'val': val_total,
            'test': test_total,
            '캡적용': '✓' if train_raw > MAX_PER_CLASS else '',
        })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))


# ─────────────────────────────────────────
# 2. 클래스(risk)별 총량 — 균형 확인
# ─────────────────────────────────────────
print()
print("="*110)
print("【2】 캡 적용 후 클래스(risk)별 총량 — 클래스 불균형 확인")
print("="*110)

risk_names = {0:'정상', 1:'초기', 2:'중기', 3:'말기'}
class_summary = res_df.groupby('risk').agg(
    train_원본=('train_원본','sum'),
    train_캡후=('train_캡후','sum'),
    val=('val','sum'),
    test=('test','sum'),
).reset_index()
class_summary['risk명'] = class_summary['risk'].map(risk_names)
class_summary = class_summary[['risk','risk명','train_원본','train_캡후','val','test']]
print(class_summary.to_string(index=False))

# 비율 비교
total_capped = class_summary['train_캡후'].sum()
print()
print("[캡 후 클래스 비율]")
for _, row in class_summary.iterrows():
    print(f"  {row['risk명']}: {row['train_캡후']:>6,} ({row['train_캡후']/total_capped*100:5.1f}%)")


# ─────────────────────────────────────────
# 3. 그룹(작물×환경)별 총량 — 그룹 균형 확인
# ─────────────────────────────────────────
print()
print("="*110)
print("【3】 캡 적용 후 그룹(작물×환경)별 총량 — 그룹 불균형 확인")
print("="*110)

group_summary = res_df.groupby(['env','crop']).agg(
    train_원본=('train_원본','sum'),
    train_캡후=('train_캡후','sum'),
    val=('val','sum'),
    test=('test','sum'),
).reset_index().sort_values('train_캡후', ascending=False)
print(group_summary.to_string(index=False))

print()
print(f"[그룹별 train 캡후 통계]")
print(f"  최대: {group_summary['train_캡후'].max():,} (그룹별 균등 16000이 캡 최대)")
print(f"  최소: {group_summary['train_캡후'].min():,}")
print(f"  중앙값: {int(group_summary['train_캡후'].median()):,}")


# ─────────────────────────────────────────
# 4. 전체 합계 및 비율
# ─────────────────────────────────────────
print()
print("="*110)
print("【4】 학습 12그룹 전체 합계")
print("="*110)
total_train_raw = res_df['train_원본'].sum()
total_train_cap = res_df['train_캡후'].sum()
total_val = res_df['val'].sum()
total_test = res_df['test'].sum()
total_all = total_train_cap + total_val + total_test
print(f"train (캡 미적용): {total_train_raw:>8,}")
print(f"train (캡 적용)  : {total_train_cap:>8,}   ← 실제 사용")
print(f"val              : {total_val:>8,}")
print(f"test             : {total_test:>8,}")
print(f"합계             : {total_all:>8,}")
print()
print(f"실제 비율: train {total_train_cap/total_all*100:.1f}% / val {total_val/total_all*100:.1f}% / test {total_test/total_all*100:.1f}%")


# ─────────────────────────────────────────
# 5. 그룹×클래스 가중치 계산 (WeightedRandomSampler용)
# ─────────────────────────────────────────
print()
print("="*110)
print("【5】 그룹×클래스 가중치 분석 (캡 적용 후 train 기준)")
print("="*110)

# weight = 1 / (그룹수 × 그룹내 클래스수)
# 정규화해서 출력
n_groups = 12
res_df['weight_raw'] = 1.0 / (n_groups * res_df['train_캡후'].clip(lower=1))
max_w = res_df['weight_raw'].max()
min_w = res_df['weight_raw'].min()
res_df['weight_정규화'] = res_df['weight_raw'] / min_w

# 가중치 분포
print(f"\n[가중치 범위]")
print(f"  최소 가중치: 1.00 (가장 많은 클래스, train_캡후 = {res_df.loc[res_df['weight_raw'].idxmin(), 'train_캡후']:,})")
print(f"  최대 가중치: {max_w/min_w:.2f}x (가장 적은 클래스)")

# 가중치 높은 케이스 (보강이 가장 많이 일어나는 케이스)
print(f"\n[가중치 상위 5개 — 학습 시 가장 자주 sampling]")
top_w = res_df.nlargest(5, 'weight_정규화')[['env','crop','risk','train_캡후','weight_정규화']]
top_w['risk명'] = top_w['risk'].map(risk_names)
print(top_w.to_string(index=False))


# ─────────────────────────────────────────
# 6. Held-out 6그룹 분석
# ─────────────────────────────────────────
print()
print("="*110)
print("【6】 Held-out 6그룹 (개방형 일반화 평가용) — train split 사용 가능 분량")
print("="*110)

ho_results = []
for env, crop in HELDOUT_GROUPS:
    g = train_df[(train_df['env']==env) & (train_df['crop_folder']==crop)]
    for risk in [0,1,2,3]:
        c = g[g['risk']==risk]
        orig = (~c['is_aug']).sum()
        aug = c['is_aug'].sum()
        ho_results.append({
            'env': env, 'crop': crop, 'risk': risk,
            'risk명': risk_names[risk],
            '원본': orig, '증강': aug, '총합': orig+aug,
        })

ho_df = pd.DataFrame(ho_results)
print(ho_df.to_string(index=False))

print()
print("[Held-out 그룹별 합계]")
ho_group = ho_df.groupby(['env','crop'])['총합'].sum().reset_index().sort_values('총합', ascending=False)
print(ho_group.to_string(index=False))
print(f"\nHeld-out 6그룹 총량: {ho_df['총합'].sum():,}장")


# ─────────────────────────────────────────
# 7. 전처리 후 저장 용량 추정
# ─────────────────────────────────────────
print()
print("="*110)
print("【7】 전처리 후 저장 용량 추정")
print("="*110)

# 512x512 JPEG 품질 85 가정: 평균 약 50KB
KB_PER_IMAGE = 50

total_images = total_train_cap + total_val + total_test + ho_df['총합'].sum()
total_size_gb = total_images * KB_PER_IMAGE / (1024 * 1024)

print(f"학습 데이터: {total_train_cap + total_val + total_test:,}장")
print(f"Held-out  : {ho_df['총합'].sum():,}장")
print(f"전체      : {total_images:,}장")
print(f"512x512 JPEG 평균 50KB 가정 → 약 {total_size_gb:.1f} GB")
print(f"구글 드라이브 3TB 중 약 {total_size_gb/3000*100:.2f}% 사용")


# ─────────────────────────────────────────
# 8. 우려사항 점검
# ─────────────────────────────────────────
print()
print("="*110)
print("【8】 우려사항 점검")
print("="*110)

# 8-1. val/test 최소 양
min_val = res_df['val'].min()
min_val_row = res_df.loc[res_df['val'].idxmin()]
print(f"\n[val 최소 케이스]")
print(f"  {min_val_row['env']} {min_val_row['crop']} {risk_names[min_val_row['risk']]}: val={min_val}, test={min_val_row['test']}")
print(f"  → val/test 평가 신뢰도: {'양호 (300+)' if min_val >= 300 else '주의 (300 미만)'}")

# 8-2. 캡 적용된 케이스 수
n_capped = (res_df['train_원본'] > MAX_PER_CLASS).sum()
print(f"\n[캡 적용 케이스]")
print(f"  전체 48개 (12그룹 × 4클래스) 중 {n_capped}개가 캡에 걸림 ({n_capped/48*100:.0f}%)")
print(f"  → 캡 적용 안 된 케이스는 양 부족 가능성, 가중치로 보완")

# 8-3. 가중치 불균형
weight_ratio = max_w / min_w
print(f"\n[가중치 불균형]")
print(f"  최대/최소 가중치 비율: {weight_ratio:.1f}x")
print(f"  → 학습 시 sampling 빈도 차이 (감당 가능 범위: 10x 이하)")
print(f"  → {'양호' if weight_ratio <= 10 else '주의'}")

# 8-4. 정상 vs 질병 비율
normal = class_summary[class_summary['risk']==0]['train_캡후'].iloc[0]
disease = class_summary[class_summary['risk']!=0]['train_캡후'].sum()
print(f"\n[정상 vs 질병 비율 (train 캡후)]")
print(f"  정상: {normal:,}장")
print(f"  질병: {disease:,}장 (초기+중기+말기)")
print(f"  비율: 1 : {disease/normal:.2f}")
print(f"  → {'정상 압도 X' if disease >= normal*0.9 else '여전히 정상 압도'}")

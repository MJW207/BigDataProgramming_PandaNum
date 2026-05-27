"""
D-1 + 캡 5000 — 작물별 상세 분석
각 작물 내부의 클래스 불균형 살펴봄
"""

import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

MAX_PER_CLASS = 5000

df = pd.read_csv('label_eda_full.csv')

TRAIN_GROUPS = [
    ('시설','05.상추'), ('시설','02.고추'), ('시설','09.쥬키니호박'),
    ('시설','04.딸기'), ('시설','11.토마토'), ('시설','03.단호박'),
    ('노지','09.파'), ('노지','03.배추'), ('노지','01.고추'),
    ('노지','04.애호박'), ('노지','06.오이'), ('노지','08.콩'),
]

train_df = df[df['split']=='train']
risk_names = {0:'정상', 1:'초기', 2:'중기', 3:'말기'}

print("="*110)
print(f"D-1 + 캡 {MAX_PER_CLASS} 기준 작물별 상세 분석")
print("="*110)

all_results = []
for env, crop in TRAIN_GROUPS:
    print(f"\n┌{'─'*108}┐")
    print(f"│ [{env}] {crop:<20s}                                                                              │")
    print(f"├{'─'*108}┤")

    g = train_df[(train_df['env']==env) & (train_df['crop_folder']==crop)]
    crop_data = []
    for risk in [0,1,2,3]:
        c = g[g['risk']==risk]
        orig = (~c['is_aug']).sum()
        aug = c['is_aug'].sum()

        # D-1 분할
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
        train_capped = min(train_raw, MAX_PER_CLASS)
        val_total = val_o + val_a
        test_total = test_o + test_a

        crop_data.append({
            'risk': risk, 'risk명': risk_names[risk],
            '원본': orig, '증강': aug,
            'train_raw': train_raw,
            'train_캡후': train_capped,
            'val': val_total, 'test': test_total,
            '캡적용': '✓' if train_raw > MAX_PER_CLASS else '',
        })
        all_results.append({
            'env': env, 'crop': crop, **crop_data[-1]
        })

    cdf = pd.DataFrame(crop_data)

    # 출력 정렬
    for _, row in cdf.iterrows():
        cap_mark = row['캡적용']
        print(f"│ {row['risk명']:4s} | 원본 {row['원본']:>6,} + 증강 {row['증강']:>6,} = {row['원본']+row['증강']:>6,} | "
              f"train {row['train_캡후']:>5,} {cap_mark:>1s} | val {row['val']:>5,} | test {row['test']:>5,} │")

    # 작물 내 불균형 분석
    min_cls = cdf['train_캡후'].min()
    max_cls = cdf['train_캡후'].max()
    min_class_name = cdf.loc[cdf['train_캡후'].idxmin(), 'risk명']
    max_class_name = cdf.loc[cdf['train_캡후'].idxmax(), 'risk명']
    imbalance = max_cls / min_cls

    print(f"├{'─'*108}┤")
    print(f"│ train 캡후 최소: {min_cls:,}장 ({min_class_name})  /  최대: {max_cls:,}장 ({max_class_name})  /  불균형: {imbalance:.2f}x │")
    print(f"│ 작물 내 가중치 보강 비율 (캡 5000 기준): 최소 {min_cls/5000:.2f}x → 최대 {5000/min_cls:.2f}x sampling{'  '*15} │")

    # 평가 가능성
    val_min = cdf['val'].min()
    test_min = cdf['test'].min()
    val_min_class = cdf.loc[cdf['val'].idxmin(), 'risk명']
    print(f"│ val 최소: {val_min}장 ({val_min_class})  /  test 최소: {test_min}장  /  평가신뢰도: "
          f"{'양호' if val_min >= 300 else '주의' if val_min >= 100 else '위험'}                              │")
    print(f"└{'─'*108}┘")

# ─────────────────────────────────────────
# 종합 — 가장 우려되는 케이스 추출
# ─────────────────────────────────────────
res_df = pd.DataFrame(all_results)

print()
print("="*110)
print("종합: 가장 우려되는 케이스 TOP 10 (train 캡후 양이 적은 순)")
print("="*110)
worry = res_df.nsmallest(10, 'train_캡후')[['env','crop','risk명','원본','증강','train_캡후','val','test']]
print(worry.to_string(index=False))

print()
print("="*110)
print("작물별 내부 불균형 비교 (작물 내 max/min 비율)")
print("="*110)
imbalance_df = res_df.groupby(['env','crop']).agg(
    최소클래스=('train_캡후','min'),
    최대클래스=('train_캡후','max'),
).reset_index()
imbalance_df['내부불균형'] = (imbalance_df['최대클래스'] / imbalance_df['최소클래스']).round(2)
imbalance_df = imbalance_df.sort_values('내부불균형', ascending=False)
print(imbalance_df.to_string(index=False))


print()
print("="*110)
print("전체 합계 (캡 5000)")
print("="*110)
print(f"train (캡 미적용): {res_df['train_raw'].sum():,}")
print(f"train (캡 적용)  : {res_df['train_캡후'].sum():,}")
print(f"val              : {res_df['val'].sum():,}")
print(f"test             : {res_df['test'].sum():,}")
total = res_df['train_캡후'].sum() + res_df['val'].sum() + res_df['test'].sum()
print(f"비율: train {res_df['train_캡후'].sum()/total*100:.1f}% / val {res_df['val'].sum()/total*100:.1f}% / test {res_df['test'].sum()/total*100:.1f}%")

# 클래스별 합계
print()
print("[클래스(risk)별 합계 — 캡 적용 후]")
class_sum = res_df.groupby('risk명').agg(
    train=('train_캡후','sum'),
    val=('val','sum'),
    test=('test','sum'),
).reindex(['정상','초기','중기','말기']).reset_index()
print(class_sum.to_string(index=False))
total_train = class_sum['train'].sum()
for _, row in class_sum.iterrows():
    print(f"  {row['risk명']}: {row['train']:,} ({row['train']/total_train*100:.1f}%)")

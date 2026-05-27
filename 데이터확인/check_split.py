import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('label_eda_full.csv')

# 학습 12그룹만 필터
train_groups = [
    ('시설','05.상추'), ('시설','02.고추'), ('시설','09.쥬키니호박'),
    ('시설','04.딸기'), ('시설','11.토마토'), ('시설','03.단호박'),
    ('노지','09.파'), ('노지','03.배추'), ('노지','01.고추'),
    ('노지','04.애호박'), ('노지','06.오이'), ('노지','08.콩'),
]

# train split만 (val은 AI Hub Validation이라 매우 작음)
train_df = df[df['split']=='train'].copy()

print("="*100)
print("그룹×클래스별 원본 vs 증강 분포 (Training 데이터)")
print("="*100)

results = []
for env, crop in train_groups:
    g = train_df[(train_df['env']==env) & (train_df['crop_folder']==crop)]
    for risk in [0,1,2,3]:
        c = g[g['risk']==risk]
        orig = (~c['is_aug']).sum()
        aug = c['is_aug'].sum()
        total = orig + aug

        # D-1: 원본 단위 70:15:15, 증강도 같은 비율로 따라감
        train_o = int(orig * 0.70)
        val_o   = int(orig * 0.15)
        test_o  = orig - train_o - val_o

        # 증강은 원본 비율 그대로 따라감 (증강이 원본별 균등 분포 가정)
        if orig > 0:
            train_a = int(aug * train_o / orig)
            val_a   = int(aug * val_o / orig)
            test_a  = aug - train_a - val_a
        else:
            train_a = val_a = test_a = 0

        results.append({
            'env': env, 'crop': crop, 'risk': risk,
            '원본': orig, '증강': aug,
            'D1_train': train_o + train_a,
            'D1_val':   val_o + val_a,
            'D1_test':  test_o + test_a,
        })

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

print()
print("="*100)
print("D-2 옵션에서 문제 클래스 (val/test 원본 < 100장)")
print("="*100)
problem = result_df[(result_df['D1_val'] < 100) | (result_df['D1_test'] < 100)]
print(problem.to_string(index=False))

print()
print("="*100)
print("D-2 옵션 전체 합계")
print("="*100)
print(f"전체 train: {result_df['D1_train'].sum():,}")
print(f"전체 val:   {result_df['D1_val'].sum():,}")
print(f"전체 test:  {result_df['D1_test'].sum():,}")
print(f"실제 비율: train {result_df['D1_train'].sum()/(result_df['D1_train'].sum()+result_df['D1_val'].sum()+result_df['D1_test'].sum())*100:.1f}% / "
      f"val {result_df['D1_val'].sum()/(result_df['D1_train'].sum()+result_df['D1_val'].sum()+result_df['D1_test'].sum())*100:.1f}% / "
      f"test {result_df['D1_test'].sum()/(result_df['D1_train'].sum()+result_df['D1_val'].sum()+result_df['D1_test'].sum())*100:.1f}%")

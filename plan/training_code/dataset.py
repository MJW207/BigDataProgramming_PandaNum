"""
dataset.py — manifest_all.csv 기반 데이터셋

■ 설계 기반: project_design.md §3.5, §3.6, §3.7

■ 핵심 특징
  - manifest_all.csv 로 이미지 경로/라벨 관리
  - D-1 분할(원본 ID 기준) 결과를 그대로 사용
  - 그룹×클래스 역빈도 WeightedRandomSampler (작물×환경×risk별 균형)
  - num_samples로 에폭당 샘플 수 제어 (학습 시간 통제)

■ split 구분
  train          → 학습 (12그룹 train_group)
  val            → 검증 (12그룹 train_group)
  test           → 테스트 (12그룹 train_group)
  external_val   → AI Hub Validation (12그룹, 외부 검증)
  heldout        → 개방형 일반화 평가 (6그룹 heldout_group)
  heldout_external → AI Hub Validation (held-out 6그룹)

■ 이미지 경로 구조
  DATA_ROOT/
    facility_02_고추/images/img_0000001.jpg
    outdoor_03_배추/images/img_0000001.jpg
    ...
"""

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────

RISK_NAMES = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}
NUM_CLASSES = 4
N_TRAIN_GROUPS = 12   # 학습 그룹 수 (가중치 계산에 사용)

# EfficientNet 버전별 입력 크기 (project_design.md §3.4)
EFFICIENTNET_INPUT_SIZE = {
    "B0": 224,
    "B3": 300,
    "B4": 380,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────
# 헬퍼: manifest 로드
# ─────────────────────────────────────────

def load_manifest(
    manifest_path: str,
    split: Optional[str] = None,
    group_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    manifest_all.csv 로드 후 필터링.

    Args:
        manifest_path : manifest_all.csv 경로
        split         : 'train' | 'val' | 'test' | 'heldout' | 'external_val' | None(전체)
        group_type    : 'train_group' | 'heldout_group' | None(전체)

    Returns:
        필터링된 DataFrame
    """
    df = pd.read_csv(manifest_path)
    df['is_aug'] = df['is_aug'].map({'True': True, 'False': False,
                                     True: True, False: False})
    df['risk'] = df['risk'].astype(int)

    if split is not None:
        df = df[df['split'] == split]
    if group_type is not None:
        df = df[df['group_type'] == group_type]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    """
    manifest 행 기반 Dataset.

    Args:
        df         : load_manifest() 반환 DataFrame (이미 필터링된 상태)
        data_root  : 이미지 루트 폴더 (group_id/images/xxx.jpg 구조)
        transform  : torchvision transform
    """

    def __init__(self, df: pd.DataFrame, data_root: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.data_root / row['group_id'] / 'images' / row['file']

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (512, 512), color=(114, 114, 114))

        if self.transform:
            img = self.transform(img)

        return img, int(row['risk'])

    def risk_labels(self) -> List[int]:
        return self.df['risk'].tolist()

    def group_ids(self) -> List[str]:
        return self.df['group_id'].tolist()


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────

def get_transforms(
    mode: str,
    efficientnet_ver: str = "B0",
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
) -> transforms.Compose:
    """
    mode: 'train' | 'val'
    efficientnet_ver: 'B0' | 'B3' | 'B4'
    """
    size = EFFICIENTNET_INPUT_SIZE.get(efficientnet_ver, 224)

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:   # val / test / heldout
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────
# WeightedRandomSampler (그룹×클래스 역빈도)
# ─────────────────────────────────────────

def make_weighted_sampler(
    df: pd.DataFrame,
    num_samples: int = 40_000,
    n_groups: int = N_TRAIN_GROUPS,
) -> WeightedRandomSampler:
    """
    그룹×클래스 역빈도 가중치 계산 (project_design.md §3.5).

    weight_i = 1.0 / (n_groups × count(env, crop_folder, risk))

    Args:
        df          : 학습 split DataFrame
        num_samples : 에폭당 샘플 수 (학습 시간 제어)
        n_groups    : 학습 그룹 수 (12)
    """
    # (env, crop_folder, risk) 별 카운트
    group_class_counts = (
        df.groupby(['env', 'crop_folder', 'risk'])
        .size()
        .to_dict()
    )

    weights = []
    for _, row in df.iterrows():
        key = (row['env'], row['crop_folder'], int(row['risk']))
        cnt = group_class_counts.get(key, 1)
        w   = 1.0 / (n_groups * cnt)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )


# ─────────────────────────────────────────
# DataLoader 빌더
# ─────────────────────────────────────────

def build_dataloaders(
    manifest_path: str,
    data_root: str,
    efficientnet_ver: str = "B0",
    batch_size: int = 64,
    num_workers: int = 4,
    num_samples: int = 40_000,
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
) -> dict:
    """
    학습/검증/테스트 DataLoader 반환.

    Returns:
        {
            'train': DataLoader,   # WeightedRandomSampler 적용
            'val'  : DataLoader,
            'test' : DataLoader,
            'external_val': DataLoader,
        }
    """
    train_df = load_manifest(manifest_path, split='train',        group_type='train_group')
    val_df   = load_manifest(manifest_path, split='val',          group_type='train_group')
    test_df  = load_manifest(manifest_path, split='test',         group_type='train_group')
    ext_df   = load_manifest(manifest_path, split='external_val', group_type='train_group')

    tf_train = get_transforms('train', efficientnet_ver, mean, std)
    tf_eval  = get_transforms('val',   efficientnet_ver, mean, std)

    train_ds   = CropDiseaseDataset(train_df, data_root, tf_train)
    val_ds     = CropDiseaseDataset(val_df,   data_root, tf_eval)
    test_ds    = CropDiseaseDataset(test_df,  data_root, tf_eval)
    ext_ds     = CropDiseaseDataset(ext_df,   data_root, tf_eval)

    sampler = make_weighted_sampler(train_df, num_samples=num_samples)
    common  = dict(num_workers=num_workers, pin_memory=(num_workers > 0))

    return {
        'train': DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, drop_last=True, **common,
        ),
        'val' : DataLoader(val_ds,  batch_size=batch_size, shuffle=False, **common),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
        'external_val': DataLoader(ext_ds, batch_size=batch_size, shuffle=False, **common),
    }


def build_heldout_loaders(
    manifest_path: str,
    data_root: str,
    efficientnet_ver: str = "B0",
    batch_size: int = 64,
    num_workers: int = 4,
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
) -> dict:
    """
    개방형 일반화 평가용 held-out DataLoader.

    Returns:
        {group_id: DataLoader, ...}  — 그룹별 개별 로더
    """
    ho_df   = load_manifest(manifest_path, split='heldout', group_type='heldout_group')
    tf_eval = get_transforms('val', efficientnet_ver, mean, std)
    common  = dict(num_workers=num_workers, pin_memory=(num_workers > 0))

    loaders = {}
    for gid in ho_df['group_id'].unique():
        g_df = ho_df[ho_df['group_id'] == gid]
        ds   = CropDiseaseDataset(g_df, data_root, tf_eval)
        loaders[gid] = DataLoader(ds, batch_size=batch_size, shuffle=False, **common)
    return loaders


# ─────────────────────────────────────────
# 분포 출력 유틸
# ─────────────────────────────────────────

def print_distribution(df: pd.DataFrame, label: str = ""):
    """split별 risk 분포 출력."""
    print(f"\n{'─'*60}")
    print(f"  {label}  (전체 {len(df):,}행)")
    print(f"{'─'*60}")
    for split in df['split'].unique():
        sub = df[df['split'] == split]
        risk_cnt = sub['risk'].value_counts().sort_index().to_dict()
        named = {RISK_NAMES[k]: v for k, v in risk_cnt.items()}
        print(f"  {split:20s}: {len(sub):>6,}행  {named}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────
# 채널 통계 계산
# ─────────────────────────────────────────

def compute_dataset_stats(
    manifest_path: str,
    data_root: str,
    num_workers: int = 4,
    batch_size: int = 128,
) -> Tuple[List[float], List[float]]:
    """
    학습 데이터의 채널별 mean/std 계산.
    1회 실행 후 결과 저장하여 재사용 권장.
    """
    df = load_manifest(manifest_path, split='train', group_type='train_group')
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds     = CropDiseaseDataset(df, data_root, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0
    for imgs, _ in loader:
        b     = imgs.size(0)
        mean += imgs.mean(dim=[0, 2, 3]) * b
        std  += imgs.std(dim=[0, 2, 3])  * b
        n    += b
    mean /= n
    std  /= n

    print(f"mean: {mean.tolist()}")
    print(f"std : {std.tolist()}")
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    # 빠른 동작 확인
    import sys
    manifest = sys.argv[1] if len(sys.argv) > 1 else "manifest_all.csv"
    data_root = sys.argv[2] if len(sys.argv) > 2 else "/workspace/data"

    df = load_manifest(manifest)
    print_distribution(df, "전체 manifest")

    train_df = load_manifest(manifest, split='train', group_type='train_group')
    print(f"\ntrain 샘플: {len(train_df):,}")
    sampler = make_weighted_sampler(train_df, num_samples=40_000)
    print(f"WeightedRandomSampler num_samples=40,000 생성 완료")

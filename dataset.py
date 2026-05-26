"""
dataset.py — 작물 질병 진행단계 분류 데이터셋

■ 변경사항 (기존 대비)
  - 기존: data/facility/train/, data/facility/val/ 각각 지정
  - 변경: data/facility/, data/outdoor/ 루트 지정 후 코드에서 70:15:15 분할
  - load_all_samples()  : 루트에서 전체 샘플 로드
  - split_samples()     : stratify 기반 70:15:15 분할
  - build_dataloaders() : 분할된 샘플 리스트를 직접 받아 로더 구성

■ Risk 라벨 매핑
  annotations.risk: 0=정상, 1=초기, 2=중기, 3=말기

■ 새 디렉터리 구조
  data/facility/
    images/  ← 전체 시설 이미지
    labels/  ← 전체 시설 라벨
  data/outdoor/
    images/
    labels/
"""

import json
from pathlib import Path
from typing import Optional, Set, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms


# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────

RISK_CLASSES   = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}
NUM_CLASSES    = len(RISK_CLASSES)
IMAGE_SIZE     = 224
IMG_EXTENSIONS = [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]
DEFAULT_CROP_FILTER: Set[int] = {2, 7, 9, 11}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────
# 라벨 파싱
# ─────────────────────────────────────────

def parse_risk(ann: dict) -> int:
    risk = int(ann.get("risk", 0))
    return risk if risk in RISK_CLASSES else 0


# ─────────────────────────────────────────
# 전체 샘플 로드 (분할 전)
# ─────────────────────────────────────────

def load_all_samples(
    root: str,
    crop_filter: Optional[Set[int]] = DEFAULT_CROP_FILTER,
) -> List[dict]:
    """
    루트 디렉터리(images/ + labels/)에서 전체 샘플을 리스트로 반환.
    분할은 split_samples()에서 수행.

    Args:
        root        : 데이터 루트 (images/, labels/ 포함)
        crop_filter : 허용 작물 코드 집합. None=전체
    Returns:
        samples: [{"img_path", "risk_label", "crop", "env"}, ...]
    """
    root     = Path(root)
    img_dir  = root / "images"
    lbl_dir  = root / "labels"
    env_name = "노지" if "outdoor" in str(root).lower() else "시설"

    if not lbl_dir.exists():
        raise FileNotFoundError(f"labels 디렉터리 없음: {lbl_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"images 디렉터리 없음: {img_dir}")

    samples      = []
    skipped_crop = skipped_img = 0

    for json_path in sorted(lbl_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        ann  = data.get("annotations", {})
        desc = data.get("description", {})

        crop = int(ann.get("crop", -1))
        if crop_filter is not None and crop not in crop_filter:
            skipped_crop += 1
            continue

        stem     = Path(desc.get("image", "")).stem
        img_path = None
        for ext in IMG_EXTENSIONS:
            p = img_dir / (stem + ext)
            if p.exists():
                img_path = p
                break

        if img_path is None:
            skipped_img += 1
            continue

        samples.append({
            "img_path":   img_path,
            "risk_label": parse_risk(ann),
            "crop":       crop,
            "env":        env_name,
        })

    # 분포 출력
    cnt = {k: 0 for k in RISK_CLASSES}
    for s in samples:
        cnt[s["risk_label"]] += 1

    print(f"[{env_name}] 전체 {len(samples):,}개 로드 "
          f"(작물필터제외:{skipped_crop} / 이미지없음:{skipped_img})", flush=True)
    print(f"  Risk 분포: { {RISK_CLASSES[k]: v for k, v in cnt.items()} }",
          flush=True)
    return samples


# ─────────────────────────────────────────
# 70:15:15 분할
# ─────────────────────────────────────────

def split_samples(
    samples:     List[dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    샘플 리스트를 risk_label 기준 stratify로 train/val/test 분할.

    Args:
        samples     : load_all_samples() 반환값
        train_ratio : 학습 비율 (기본 0.70)
        val_ratio   : 검증 비율 (기본 0.15) → test = 1 - train - val
        seed        : 재현성 시드
    Returns:
        train_samples, val_samples, test_samples
    """
    labels        = [s["risk_label"] for s in samples]
    test_val_ratio = 1.0 - train_ratio

    # 1단계: train vs (val + test)
    train_s, temp_s, _, temp_labels = train_test_split(
        samples, labels,
        test_size    = test_val_ratio,
        stratify     = labels,
        random_state = seed,
    )

    # 2단계: val vs test
    val_ratio_of_temp = val_ratio / test_val_ratio
    val_s, test_s, _, _ = train_test_split(
        temp_s, temp_labels,
        test_size    = 1.0 - val_ratio_of_temp,
        stratify     = temp_labels,
        random_state = seed,
    )

    return train_s, val_s, test_s


def print_split_info(
    train_s: List[dict],
    val_s:   List[dict],
    test_s:  List[dict],
):
    """분할 결과 분포 출력."""
    total = len(train_s) + len(val_s) + len(test_s)
    print(f"\n{'─'*60}", flush=True)
    print(f"  데이터 분할 결과 (전체 {total:,}개)", flush=True)
    print(f"{'─'*60}", flush=True)
    for name, slist in [("Train", train_s), ("Val", val_s), ("Test", test_s)]:
        cnt = {RISK_CLASSES[k]: sum(1 for x in slist if x["risk_label"] == k)
               for k in RISK_CLASSES}
        pct = len(slist) / total * 100
        print(f"  {name:5s}: {len(slist):6,}개 ({pct:4.1f}%)  {cnt}", flush=True)
    print(f"{'─'*60}\n", flush=True)


# ─────────────────────────────────────────
# Dataset (샘플 리스트 기반)
# ─────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    """
    샘플 리스트를 받아 Dataset을 구성.
    load_all_samples() + split_samples() 후 전달.
    """

    def __init__(self,
                 samples:   List[dict],
                 transform = None,
                 mode:      str = "train"):
        self.samples   = samples
        self.transform = transform
        self.mode      = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img    = cv2.imread(str(sample["img_path"]))
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, torch.tensor(sample["risk_label"], dtype=torch.long)


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────

def get_transforms(mode: str, mean=_MEAN, std=_STD) -> transforms.Compose:
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────
# WeightedRandomSampler
# ─────────────────────────────────────────

def make_weighted_sampler(samples: List[dict]) -> WeightedRandomSampler:
    labels    = [s["risk_label"] for s in samples]
    class_cnt = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_wt  = 1.0 / np.maximum(class_cnt, 1)
    sample_wt = [class_wt[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_wt, num_samples=len(sample_wt), replacement=True
    )


# ─────────────────────────────────────────
# 데이터셋 통계 계산
# ─────────────────────────────────────────

def compute_dataset_stats(
    train_samples: List[dict],
    num_workers:   int = 4,
    batch_size:    int = 64,
) -> Tuple[List[float], List[float]]:
    """
    학습 샘플의 채널별 mean/std 계산 (Normalize 없이).
    1회 실행 후 JSON으로 저장하여 재사용 권장.
    """
    raw_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    ds     = CropDiseaseDataset(train_samples, raw_tf, "stats")
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

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
    print(f"데이터셋 통계 — mean: {mean.tolist()}", flush=True)
    print(f"데이터셋 통계 — std : {std.tolist()}",  flush=True)
    return mean.tolist(), std.tolist()


# ─────────────────────────────────────────
# DataLoader 빌더
# ─────────────────────────────────────────

def build_dataloaders(
    train_samples:        List[dict],
    val_samples:          List[dict],
    test_samples:         Optional[List[dict]] = None,
    batch_size:           int  = 32,
    num_workers:          int  = 0,
    use_weighted_sampler: bool = True,
    mean = _MEAN,
    std  = _STD,
) -> dict:
    """
    분할된 샘플 리스트를 받아 DataLoader dict 반환.

    Returns:
        {"train": ..., "val": ..., "test": ...}
        test_samples=None 이면 "test" 키 없음
    """
    train_ds = CropDiseaseDataset(train_samples, get_transforms("train", mean, std), "train")
    val_ds   = CropDiseaseDataset(val_samples,   get_transforms("val",   mean, std), "val")

    sampler = make_weighted_sampler(train_samples) if use_weighted_sampler else None
    common  = dict(num_workers=num_workers, pin_memory=(num_workers > 0))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **common
    )
    loaders = {"train": train_loader, "val": val_loader}

    if test_samples is not None:
        test_ds = CropDiseaseDataset(
            test_samples, get_transforms("val", mean, std), "test"
        )
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, **common
        )

    return loaders

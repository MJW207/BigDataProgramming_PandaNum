"""
dataset.py v6 — manifest_all.csv 기반 데이터셋 (zip 직접 읽기)

■ 데이터 흐름
  로컬 전처리 (preprocess.ipynb):
    원본 → bbox 크롭 → Letterbox 512×512 → images/ + manifest.csv → zip
  통합 (merge_manifests.ipynb):
    각 작물 zip → manifest_all.csv
  RunPod 업로드:
    zip 파일 + manifest_all.csv (압축 해제 불필요)

■ RunPod 디렉터리 구조
  /workspace/data/
    manifest_all.csv
    facility_02_고추.zip    ← zip 그대로 사용
    facility_03_단호박.zip
    outdoor_01_고추.zip
    ...

■ manifest_all.csv 컬럼
  file        : img_0000001.jpg
  env         : 시설 | 노지
  crop_folder : 02.고추 | 01.고추 등
  crop_code   : 정수 (환경별 독립)
  disease     : 정수
  risk        : 0=정상 1=초기 2=중기 3=말기
  grow        : 정수
  original_id : 원본 ID (D-1 분할 기준, 전처리에서 완료)
  is_aug      : True | False
  split       : train | val | test | heldout | external_val | heldout_external
  group_type  : train_group | heldout_group
  group_id    : facility_02_고추 (이미지 폴더명과 동일)

■ split 값 의미
  train          : 학습용 (D-1 분할, 캡 5000 적용)
  val            : 검증용 (D-1 분할, 캡 없음)
  test           : 테스트용 (D-1 분할, 캡 없음)
  heldout        : 개방형 일반화 평가용 (학습 미사용 작물)
  external_val   : AI Hub 원본 Validation split (미사용)
  heldout_external: held-out의 AI Hub Validation (미사용)
"""

import csv
import threading
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ─────────────────────────────────────────
# zip 스레드-로컬 캐시 (NUM_WORKERS 멀티프로세스 안전)
# ─────────────────────────────────────────

_zip_cache = threading.local()

def _get_zip(zip_path: str) -> zipfile.ZipFile:
    if not hasattr(_zip_cache, 'handles'):
        _zip_cache.handles = {}
    if zip_path not in _zip_cache.handles:
        _zip_cache.handles[zip_path] = zipfile.ZipFile(zip_path, 'r')
    return _zip_cache.handles[zip_path]


# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────

RISK_CLASSES = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}
NUM_CLASSES  = len(RISK_CLASSES)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

MODEL_INPUT_SIZE = {"B0": 224, "B3": 300, "B4": 380}


# ─────────────────────────────────────────
# manifest 로드
# ─────────────────────────────────────────

def load_manifest(
    manifest_path: str,
    data_root:     str,
    splits:        Set[str],
    group_type:    Optional[str] = None,
) -> List[dict]:
    """
    manifest_all.csv 에서 조건에 맞는 행을 읽어 샘플 리스트 반환.

    Args:
        manifest_path : manifest_all.csv 경로
        data_root     : 이미지 루트 (/workspace/data)
        splits        : 포함할 split 값 집합
                        예) {"train"} | {"val"} | {"test"} | {"heldout"}
        group_type    : "train_group" | "heldout_group" | None(전체)
    Returns:
        samples 리스트 (dict 형태)
    """
    data_root = Path(data_root)
    samples   = []
    skipped   = 0

    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] not in splits:
                continue
            if group_type and row["group_type"] != group_type:
                continue

            risk = int(row["risk"])
            if risk not in RISK_CLASSES:
                skipped += 1
                continue

            # zip 파일 존재 확인 (압축 해제 불필요)
            zip_path = data_root / f"{row['group_id']}.zip"
            if not zip_path.exists():
                skipped += 1
                continue

            samples.append({
                "zip_path":    str(zip_path),
                "inner_path":  f"{row['group_id']}/images/{row['file']}",
                "risk_label":  risk,
                "crop_code":   int(row["crop_code"]),
                "crop_folder": row["crop_folder"],
                "env":         row["env"],
                "group_id":    row["group_id"],
                "group_type":  row["group_type"],
                "group_key":   (row["group_id"], risk),
            })

    split_str = "+".join(sorted(splits))
    cnt_r = Counter(s["risk_label"] for s in samples)
    print(f"[{split_str}] {len(samples):,}개 로드  (스킵:{skipped})", flush=True)
    print(f"  Risk 분포: { {RISK_CLASSES[k]: cnt_r[k] for k in sorted(cnt_r)} }",
          flush=True)
    return samples


def load_all_splits(
    manifest_path: str,
    data_root:     str,
) -> Dict[str, List[dict]]:
    """
    train / val / test / heldout 를 한 번에 로드.

    Returns:
        {
          "train":   [...],   # group_type=train_group, split=train
          "val":     [...],   # group_type=train_group, split=val
          "test":    [...],   # group_type=train_group, split=test
          "heldout": [...],   # group_type=heldout_group, split=heldout
        }
    """
    configs = [
        ("train",   {"train"},   "train_group"),
        ("val",     {"val"},     "train_group"),
        ("test",    {"test"},    "train_group"),
        ("heldout", {"heldout"}, "heldout_group"),
    ]
    return {
        name: load_manifest(manifest_path, data_root, splits, gtype)
        for name, splits, gtype in configs
    }


def print_split_summary(splits: Dict[str, List[dict]]):
    total = sum(len(v) for v in splits.values())
    print(f"\n{'─'*65}", flush=True)
    print(f"  데이터셋 요약 (전체 {total:,}개)", flush=True)
    print(f"{'─'*65}", flush=True)
    for name, slist in splits.items():
        cnt = {RISK_CLASSES[k]: sum(1 for s in slist if s["risk_label"] == k)
               for k in RISK_CLASSES}
        pct = len(slist) / total * 100 if total else 0
        print(f"  {name:8s}: {len(slist):7,}개 ({pct:4.1f}%)  {cnt}", flush=True)
    print(f"{'─'*65}\n", flush=True)


def print_group_distribution(samples: List[dict], title: str = "그룹 분포"):
    """(group_id, risk) 단위 세부 분포."""
    cnt = Counter(s["group_key"] for s in samples)
    print(f"\n  [{title}]", flush=True)
    for (gid, risk), n in sorted(cnt.items()):
        print(f"    {gid:30s} {RISK_CLASSES[risk]:3s}: {n:6,}", flush=True)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    def __init__(self, samples: List[dict], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        lbl = torch.tensor(s["risk_label"], dtype=torch.long)

        # 스레드-로컬 캐시 ZipFile로 읽기
        # KeyError 발생 시 ZipFile 재오픈 후 재시도
        try:
            img_bytes = _get_zip(s["zip_path"]).read(s["inner_path"])
        except KeyError:
            # 캐시 ZipFile 객체 제거 후 재오픈
            if hasattr(_zip_cache, 'handles'):
                _zip_cache.handles.pop(s["zip_path"], None)
            img_bytes = _get_zip(s["zip_path"]).read(s["inner_path"])
        except Exception:
            img_bytes = None

        if img_bytes is not None:
            img_cv = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_cv is not None:
                img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            else:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, lbl


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────

def get_transforms(mode: str, mean=_MEAN, std=_STD,
                   model_ver: str = "B0") -> transforms.Compose:
    """
    Args:
        mode      : "train" | "val"  (val/test/heldout 모두 "val" 사용)
        model_ver : "B0"(224) | "B3"(300) | "B4"(380)
    """
    size = MODEL_INPUT_SIZE.get(model_ver, 224)
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────
# 그룹×클래스 역빈도 WeightedRandomSampler
# ─────────────────────────────────────────

def make_group_weighted_sampler(
    samples:     List[dict],
    num_samples: int = 40_000,
) -> WeightedRandomSampler:
    """
    (group_id, risk) 조합 단위 역빈도 가중치.

    group_id = facility_02_고추 처럼 환경 정보가 포함되어 있어
    crop_code 환경별 중복 문제 없이 그룹을 정확히 구분.

    weight_i = 1 / count(group_key_i)
      → 딸기 정상(19,679) → 낮은 가중치
      → 단호박 정상(2,751) → 높은 가중치
      → 12그룹이 에폭당 동등하게 기여

    Args:
        num_samples : 에폭당 샘플 수 (학습 시간 제어)
    """
    group_cnt = Counter(s["group_key"] for s in samples)
    weights   = [1.0 / group_cnt[s["group_key"]] for s in samples]

    print(f"\n  [WeightedRandomSampler] 그룹×클래스 역빈도", flush=True)
    for (gid, risk), cnt in sorted(group_cnt.items()):
        print(f"    {gid:30s} {RISK_CLASSES[risk]:3s}: "
              f"{cnt:6,}개  w={1/cnt:.2e}", flush=True)
    print(f"  에폭당 샘플 수: {num_samples:,}", flush=True)

    return WeightedRandomSampler(
        weights=weights, num_samples=num_samples, replacement=True)


# ─────────────────────────────────────────
# 데이터셋 통계 계산
# ─────────────────────────────────────────

def compute_dataset_stats(
    train_samples: List[dict],
    model_ver:     str = "B0",
    num_workers:   int = 0,
    batch_size:    int = 64,
) -> Tuple[List[float], List[float]]:
    """학습 샘플의 채널별 mean/std 계산 (1회만 실행 후 JSON 저장 권장)."""
    size   = MODEL_INPUT_SIZE.get(model_ver, 224)
    raw_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    ds     = CropDiseaseDataset(train_samples, raw_tf)
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
    mean /= n; std /= n
    print(f"데이터셋 통계 ({model_ver}) — mean: {mean.tolist()}", flush=True)
    print(f"데이터셋 통계 ({model_ver}) — std : {std.tolist()}",  flush=True)
    return mean.tolist(), std.tolist()


# ─────────────────────────────────────────
# DataLoader 빌더
# ─────────────────────────────────────────

def build_dataloaders(
    train_samples:  List[dict],
    val_samples:    List[dict],
    test_samples:   Optional[List[dict]] = None,
    batch_size:     int  = 32,
    num_workers:    int  = 0,
    num_samples:    int  = 40_000,
    mean = _MEAN,
    std  = _STD,
    model_ver: str = "B0",
) -> dict:
    """
    Args:
        num_samples : 에폭당 샘플 수 (WeightedRandomSampler)
        model_ver   : "B0" | "B3" | "B4"

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader(있을 때만)}
    """
    train_tf = get_transforms("train", mean, std, model_ver)
    val_tf   = get_transforms("val",   mean, std, model_ver)

    sampler = make_group_weighted_sampler(train_samples, num_samples)
    common  = dict(
        num_workers    = num_workers,
        pin_memory     = (num_workers > 0),
        prefetch_factor= 4 if num_workers > 0 else None,
        persistent_workers = (num_workers > 0),
    )

    loaders = {
        "train": DataLoader(
            CropDiseaseDataset(train_samples, train_tf),
            batch_size=batch_size, sampler=sampler, drop_last=True, **common,
        ),
        "val": DataLoader(
            CropDiseaseDataset(val_samples, val_tf),
            batch_size=batch_size, shuffle=False, **common,
        ),
    }
    if test_samples is not None:
        loaders["test"] = DataLoader(
            CropDiseaseDataset(test_samples, val_tf),
            batch_size=batch_size, shuffle=False, **common,
        )
    return loaders

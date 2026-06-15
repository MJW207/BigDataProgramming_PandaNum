"""
split_utils.py — D-1 분할 (원본 ID 기준 70:15:15)

같은 원본 이미지에서 파생된 증강 이미지가 train/val/test에
분산되는 데이터 누수를 방지.

원본 ID 단위로 stratified 70:15:15 분할 후,
같은 원본의 증강은 같은 split에 자동 배치.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, SPLIT_RATIO
from label_utils import LabelInfo


def split_train_group(labels: List[LabelInfo]) -> Dict[str, str]:
    """
    학습 그룹(train_group)의 라벨들을 원본 ID 기준 70:15:15 분할.

    Args:
        labels: 한 그룹(env, crop_folder)의 모든 LabelInfo 리스트
                (Training만, AI Hub Validation 제외)

    Returns:
        {original_id: split_name} 매핑
        split_name ∈ {"train", "val", "test"}
    """
    train_r, val_r, test_r = SPLIT_RATIO
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    # 원본 ID → 대표 risk (같은 원본은 모두 같은 risk 가정)
    original_to_risk: Dict[str, int] = {}
    for l in labels:
        if l.split_aihub != "train":
            continue
        if l.original_id not in original_to_risk:
            original_to_risk[l.original_id] = l.risk

    original_ids = list(original_to_risk.keys())
    risks        = [original_to_risk[oid] for oid in original_ids]

    if len(original_ids) == 0:
        return {}

    # 1단계: train vs (val + test)
    test_val_size = val_r + test_r
    try:
        train_ids, temp_ids, _, temp_risks = train_test_split(
            original_ids, risks,
            test_size=test_val_size,
            stratify=risks,
            random_state=RANDOM_SEED,
        )
    except ValueError:
        # 한 클래스가 너무 적어서 stratify 실패 시 단순 random
        train_ids, temp_ids, _, temp_risks = train_test_split(
            original_ids, risks,
            test_size=test_val_size,
            random_state=RANDOM_SEED,
        )

    # 2단계: val vs test
    val_ratio_of_temp = val_r / test_val_size
    try:
        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_risks,
            test_size=1.0 - val_ratio_of_temp,
            stratify=temp_risks,
            random_state=RANDOM_SEED,
        )
    except ValueError:
        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_risks,
            test_size=1.0 - val_ratio_of_temp,
            random_state=RANDOM_SEED,
        )

    result: Dict[str, str] = {}
    for oid in train_ids:
        result[oid] = "train"
    for oid in val_ids:
        result[oid] = "val"
    for oid in test_ids:
        result[oid] = "test"
    return result


def assign_splits(labels: List[LabelInfo], group_type: str) -> List[Tuple[LabelInfo, str]]:
    """
    각 라벨에 split 부여.

    - train_group:
        AI Hub Training은 원본 ID 기준 70:15:15 분할
        AI Hub Validation은 별도로 "external_val"로 표시
    - heldout_group:
        모두 "heldout"으로 표시

    Returns:
        [(LabelInfo, split), ...]
    """
    results: List[Tuple[LabelInfo, str]] = []

    if group_type == "heldout_group":
        for l in labels:
            split = "heldout" if l.split_aihub == "train" else "heldout_external"
            results.append((l, split))
        return results

    # train_group: 원본 ID 기준 분할
    id_to_split = split_train_group(labels)

    for l in labels:
        if l.split_aihub == "val":
            results.append((l, "external_val"))   # AI Hub Validation = 외부 검증셋
        else:
            split = id_to_split.get(l.original_id, "train")
            results.append((l, split))

    return results


def split_distribution(splits: List[Tuple[LabelInfo, str]]) -> Dict[str, Dict]:
    """split별 분포 요약."""
    from collections import Counter

    by_split = defaultdict(list)
    for l, s in splits:
        by_split[s].append(l)

    summary = {}
    for split, lbls in by_split.items():
        risk_count = Counter(l.risk for l in lbls)
        summary[split] = {
            "total": len(lbls),
            "by_risk": dict(sorted(risk_count.items())),
            "unique_originals": len(set(l.original_id for l in lbls)),
        }
    return summary


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from label_utils import load_group_labels

    label_zip = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./deeplearning_labeling.zip")
    env = sys.argv[2] if len(sys.argv) > 2 else "시설"
    crop = sys.argv[3] if len(sys.argv) > 3 else "02.고추"

    labels = load_group_labels(label_zip, env, crop)
    splits = assign_splits(labels, "train_group")
    dist = split_distribution(splits)

    print(f"=== {env} {crop} 분할 결과 ===")
    for split, info in dist.items():
        print(f"  {split}: {info}")

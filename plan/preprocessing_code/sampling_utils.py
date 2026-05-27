"""
sampling_utils.py — MAX_PER_CLASS = 5000 캡 적용

train split에 한해 그룹×클래스별 최대 5000장 random sampling.
val/test/external_val/heldout은 캡 적용 안 함.
"""

import random
from collections import defaultdict
from typing import List, Tuple

from config import MAX_PER_CLASS, RANDOM_SEED
from label_utils import LabelInfo


def apply_cap(splits: List[Tuple[LabelInfo, str]]) -> List[Tuple[LabelInfo, str]]:
    """
    train split에만 그룹×클래스(risk)별 캡 적용.

    Args:
        splits: assign_splits() 반환값

    Returns:
        캡 적용된 [(LabelInfo, split), ...]
    """
    rng = random.Random(RANDOM_SEED)

    # train split만 risk별로 분류
    train_by_risk = defaultdict(list)
    other = []

    for l, s in splits:
        if s == "train":
            train_by_risk[l.risk].append((l, s))
        else:
            other.append((l, s))

    # 각 risk별로 캡 적용
    capped_train = []
    for risk, items in train_by_risk.items():
        if len(items) > MAX_PER_CLASS:
            # 같은 원본은 함께 sampling (누수 방지 유지) → 원본 ID 단위 sampling
            by_origin = defaultdict(list)
            for l, s in items:
                by_origin[l.original_id].append((l, s))

            origins = list(by_origin.keys())
            rng.shuffle(origins)

            collected = []
            for oid in origins:
                if len(collected) + len(by_origin[oid]) > MAX_PER_CLASS:
                    # 일부만 추가 (캡 정확히 맞추기)
                    remain = MAX_PER_CLASS - len(collected)
                    if remain > 0:
                        collected.extend(by_origin[oid][:remain])
                    break
                collected.extend(by_origin[oid])

            capped_train.extend(collected)
        else:
            capped_train.extend(items)

    return capped_train + other


def cap_summary(before: List[Tuple[LabelInfo, str]],
                after: List[Tuple[LabelInfo, str]]) -> dict:
    """캡 적용 전후 비교."""
    from collections import Counter

    def by_split_risk(items):
        d = defaultdict(lambda: Counter())
        for l, s in items:
            d[s][l.risk] += 1
        return {s: dict(c) for s, c in d.items()}

    return {
        "before": by_split_risk(before),
        "after":  by_split_risk(after),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from label_utils import load_group_labels
    from split_utils import assign_splits

    label_zip = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./deeplearning_labeling.zip")
    env = sys.argv[2] if len(sys.argv) > 2 else "시설"
    crop = sys.argv[3] if len(sys.argv) > 3 else "02.고추"

    labels = load_group_labels(label_zip, env, crop)
    splits = assign_splits(labels, "train_group")
    capped = apply_cap(splits)
    summary = cap_summary(splits, capped)

    print(f"=== {env} {crop} 캡 적용 전후 ===")
    print("[전]", summary["before"])
    print("[후]", summary["after"])

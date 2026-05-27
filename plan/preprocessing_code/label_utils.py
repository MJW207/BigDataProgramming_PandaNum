"""
label_utils.py — JSON 라벨 파싱 및 원본 ID 추출

deeplearning_labeling.zip에서 작물별 JSON을 읽어
- crop, disease, risk, points 추출
- 증강 이미지의 원본 ID 매칭 (description.original)
"""

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import LABEL_ROOT


# ─────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────

@dataclass
class LabelInfo:
    """JSON 라벨에서 추출한 정보"""
    image_filename: str       # 이미지 파일명 (확장자 포함, 예: V006_..._.jpg)
    env: str                  # "시설" or "노지"
    crop_folder: str          # 예: "02.고추"
    data_type: str            # "정상" / "질병" / "증강"
    split_aihub: str          # "train" or "val" (AI Hub 원본 split)
    crop_code: int
    disease: int
    risk: int
    grow: int
    bbox: Optional[Dict[str, int]]   # {xtl, ytl, xbr, ybr} or None
    image_height: int
    image_width: int
    is_aug: bool
    original_id: str          # 원본 ID (증강이면 description.original, 아니면 자기 자신)


# ─────────────────────────────────────────
# 파일명에서 원본 ID 추출
# ─────────────────────────────────────────

# 증강 파일명 패턴: 원본명 + "_a####.jpg"
# 예: V006_77_1_01_01_03_13_1_1929e_20201203_10016_a0000.jpg
_AUG_SUFFIX_RE = re.compile(r"_a\d+(\.\w+)$")


def extract_original_id(image_filename: str, original_from_json: Optional[str] = None) -> str:
    """
    원본 ID 추출.
    - JSON에 description.original 있으면 그걸 사용 (가장 정확)
    - 없으면 파일명에서 _a#### suffix 제거
    - 그래도 없으면 파일명 그대로 (정상/질병 원본)
    """
    if original_from_json:
        return Path(original_from_json).stem

    # _a#### 패턴 제거
    name = _AUG_SUFFIX_RE.sub(r"\1", image_filename)
    return Path(name).stem


# ─────────────────────────────────────────
# 단일 JSON 파싱
# ─────────────────────────────────────────

def parse_label_json(json_bytes: bytes, env: str, crop_folder: str,
                     data_type: str, split_aihub: str) -> Optional[LabelInfo]:
    """JSON 바이트 → LabelInfo. 실패 시 None."""
    try:
        data = json.loads(json_bytes.decode("utf-8"))
    except Exception:
        return None

    desc = data.get("description", {})
    ann  = data.get("annotations", {})
    is_aug = "augmented" in data or "_a" in desc.get("image", "")

    image_filename = desc.get("image", "")
    if not image_filename:
        return None

    # bbox
    points = ann.get("points", [])
    bbox = points[0] if points else None
    if bbox and not all(k in bbox for k in ("xtl", "ytl", "xbr", "ybr")):
        bbox = None

    original_id = extract_original_id(image_filename, desc.get("original"))

    return LabelInfo(
        image_filename=image_filename,
        env=env,
        crop_folder=crop_folder,
        data_type=data_type,
        split_aihub=split_aihub,
        crop_code=int(ann.get("crop", -1)),
        disease=int(ann.get("disease", -1)),
        risk=int(ann.get("risk", -1)),
        grow=int(ann.get("grow", -1)),
        bbox=bbox,
        image_height=int(desc.get("height", 0)),
        image_width=int(desc.get("width", 0)),
        is_aug=is_aug,
        original_id=original_id,
    )


# ─────────────────────────────────────────
# deeplearning_labeling.zip 에서 그룹별 라벨 추출
# ─────────────────────────────────────────

def load_group_labels(label_zip_path: Path, env: str, crop_folder: str) -> List[LabelInfo]:
    """
    deeplearning_labeling.zip 에서 (env, crop_folder)에 해당하는
    Training + Validation 모든 JSON 파싱.
    """
    labels: List[LabelInfo] = []
    root_prefix = LABEL_ROOT[env]

    # 매칭할 경로 패턴 (Training, Validation 둘 다)
    targets = [
        ("1.Training",   "train", "0.정상", "정상"),
        ("1.Training",   "train", "1.질병", "질병"),
        ("1.Training",   "train", "9.증강", "증강"),
        ("2.Validation", "val",   "0.정상", "정상"),
        ("2.Validation", "val",   "1.질병", "질병"),
        ("2.Validation", "val",   "9.증강", "증강"),
    ]

    with zipfile.ZipFile(label_zip_path, "r") as zf:
        for entry in zf.infolist():
            if not entry.filename.endswith(".json"):
                continue
            if root_prefix not in entry.filename:
                continue
            if f"/{crop_folder}/" not in entry.filename:
                continue

            # split / data_type 결정
            matched = None
            for subdir, split, type_token, type_name in targets:
                if f"/{subdir}/" in entry.filename and f"/{type_token}.zip/" in entry.filename:
                    matched = (split, type_name)
                    break
            if matched is None:
                continue
            split_aihub, data_type = matched

            with zf.open(entry) as f:
                json_bytes = f.read()

            info = parse_label_json(json_bytes, env, crop_folder, data_type, split_aihub)
            if info:
                labels.append(info)

    return labels


# ─────────────────────────────────────────
# 검증
# ─────────────────────────────────────────

def summarize_labels(labels: List[LabelInfo]) -> Dict:
    """라벨 통계 요약."""
    from collections import Counter

    risk_count = Counter(l.risk for l in labels)
    type_count = Counter(l.data_type for l in labels)
    split_count = Counter(l.split_aihub for l in labels)
    aug_count = Counter(l.is_aug for l in labels)
    original_ids = set(l.original_id for l in labels)

    return {
        "total": len(labels),
        "by_risk": dict(risk_count),
        "by_type": dict(type_count),
        "by_split_aihub": dict(split_count),
        "is_aug_true": aug_count.get(True, 0),
        "is_aug_false": aug_count.get(False, 0),
        "unique_originals": len(original_ids),
    }


if __name__ == "__main__":
    # 빠른 동작 확인 (가지 한 그룹)
    from pathlib import Path
    import sys

    label_zip = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./deeplearning_labeling.zip")
    if not label_zip.exists():
        print(f"❌ {label_zip} 없음. 경로 인자로 전달하세요.")
        sys.exit(1)

    labels = load_group_labels(label_zip, "시설", "01.가지")
    summary = summarize_labels(labels)
    print(f"=== 시설 01.가지 라벨 요약 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

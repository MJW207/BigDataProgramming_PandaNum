"""
config.py — 전처리 공통 설정

모든 팀원이 같은 결과를 얻도록 절대 수정 금지 (RANDOM_SEED 등).
경로 관련은 각자 자기 환경에 맞게 수정.
"""

from pathlib import Path

# ─────────────────────────────────────────
# 경로 설정 (각자 자기 환경에 맞게 수정)
# ─────────────────────────────────────────

# 원천 데이터(이미지 zip) 루트 경로
# 예: C:/data/raw/  또는 /home/user/data/raw/
RAW_DATA_ROOT = Path("./raw_data")

# 라벨링 데이터 zip 경로 (deeplearning_labeling.zip)
LABEL_ZIP_PATH = Path("./deeplearning_labeling.zip")

# 전처리 결과 저장 루트
OUTPUT_ROOT = Path("./preprocessed")


# ─────────────────────────────────────────
# 절대 수정 금지 — 표준화 값
# ─────────────────────────────────────────

RANDOM_SEED   = 42
TARGET_SIZE   = 512                  # letterbox resize 크기
JPEG_QUALITY  = 90
PADDING_RATIO = 0.30                 # bbox 30% 패딩
MAX_PER_CLASS = 5000                 # train 그룹×클래스 캡
SPLIT_RATIO   = (0.70, 0.15, 0.15)   # train/val/test
INTERPOLATION = "INTER_LANCZOS4"     # cv2.INTER_LANCZOS4

NUM_CLASSES = 4
RISK_NAMES  = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}


# ─────────────────────────────────────────
# 그룹 정의
# ─────────────────────────────────────────

# 학습 12그룹 (env, crop_folder)
TRAIN_GROUPS = [
    ("시설", "02.고추"),
    ("시설", "03.단호박"),
    ("시설", "04.딸기"),
    ("시설", "05.상추"),
    ("시설", "09.쥬키니호박"),
    ("시설", "11.토마토"),
    ("노지", "01.고추"),
    ("노지", "03.배추"),
    ("노지", "04.애호박"),
    ("노지", "06.오이"),
    ("노지", "08.콩"),
    ("노지", "09.파"),
]

# held-out 6그룹 (학습 미사용, 개방형 일반화 평가용)
HELDOUT_GROUPS = [
    ("시설", "01.가지"),
    ("시설", "06.수박"),
    ("시설", "10.참외"),
    ("노지", "05.양배추"),
    ("노지", "07.잎마름병(토마토)"),
    ("노지", "10.호박"),
]

# 완전 제외 그룹 (다운로드 금지)
EXCLUDED_GROUPS = [
    ("시설", "07.애호박"),   # 말기 0장
    ("시설", "08.오이"),     # min_class 176
    ("시설", "12.포도"),     # min_class 22
    ("노지", "02.무"),       # min_class 384
]

# env 영문 매핑 (파일명/경로용)
ENV_EN = {"시설": "facility", "노지": "outdoor"}


# ─────────────────────────────────────────
# 환경별 원천 데이터셋 키 (AI Hub)
# ─────────────────────────────────────────

DATASET_KEY = {
    "시설": 153,   # 071. 시설 작물 질병 진단
    "노지": 147,   # 073. 노지 작물 질병 진단
}


# ─────────────────────────────────────────
# zip 파일명 패턴 (원천 데이터)
# ─────────────────────────────────────────

# 다운로드 후 사용자가 정리하는 폴더 구조 예시:
#   RAW_DATA_ROOT/
#     facility/02.고추/
#       Training/
#         0.정상.zip, 1.질병.zip, 9.증강_(1).zip, ...
#       Validation/
#         0.정상.zip, 1.질병.zip, 9.증강.zip
#     outdoor/01.고추/...

SUBFOLDER_TRAIN = "Training"
SUBFOLDER_VAL   = "Validation"


# ─────────────────────────────────────────
# 라벨 zip 내부 경로 패턴
# ─────────────────────────────────────────

# deeplearning_labeling.zip 내부 경로:
#   071.시설_작물_질병_진단/01.데이터/1.Training/라벨링데이터/01.가지/0.정상.zip/...json
#   073.노지_작물_질병_진단/01.데이터/2.Validation/라벨링데이터/01.고추/9.증강.zip/...json

LABEL_ROOT = {
    "시설": "071.시설_작물_질병_진단",
    "노지": "073.노지_작물_질병_진단",
}


# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────

def all_target_groups():
    """학습 + held-out 그룹 모두 (제외는 미포함)"""
    return [(e, c, "train") for e, c in TRAIN_GROUPS] + \
           [(e, c, "heldout") for e, c in HELDOUT_GROUPS]


def group_id(env: str, crop_folder: str) -> str:
    """그룹 식별자 (파일명용). 예: facility_02_고추"""
    crop_num = crop_folder.split(".")[0]
    crop_name = crop_folder.split(".")[1] if "." in crop_folder else crop_folder
    # 특수문자 제거 (잎마름병(토마토) → 잎마름병토마토)
    crop_name = crop_name.replace("(", "").replace(")", "")
    return f"{ENV_EN[env]}_{crop_num}_{crop_name}"


def group_type_of(env: str, crop_folder: str) -> str:
    """그룹이 train 그룹인지 heldout 그룹인지"""
    if (env, crop_folder) in TRAIN_GROUPS:
        return "train_group"
    if (env, crop_folder) in HELDOUT_GROUPS:
        return "heldout_group"
    return "excluded"


if __name__ == "__main__":
    print("=== 그룹 ID 미리보기 ===")
    for env, crop in TRAIN_GROUPS + HELDOUT_GROUPS:
        print(f"  ({env}, {crop}) → {group_id(env, crop)}  [{group_type_of(env, crop)}]")

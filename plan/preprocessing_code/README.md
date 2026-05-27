# 전처리 코드 사용 가이드

작물 질병 진행단계 분류 모델 학습용 데이터 사전 전처리 코드.

**모든 팀원이 같은 결과를 얻기 위해 표준 코드를 공유합니다.**

---

## 파일 구조

```
preprocessing_code/
├── preprocess.ipynb         # 메인 실행 노트북 (각 팀원이 사용)
├── merge_manifests.ipynb    # 통합 노트북 (한 명만 실행)
├── config.py                # 경로 + 그룹 + 표준 설정
├── label_utils.py           # JSON 파싱 + 원본 ID 추출
├── crop_utils.py            # bbox 크롭 + Letterbox resize
├── split_utils.py           # D-1 분할 (원본 ID 기준 70:15:15)
├── sampling_utils.py        # 캡 5000 적용
├── requirements.txt         # 필요 패키지
├── README.md                # 이 파일
└── AI_PROMPT_TEMPLATE.md    # AI 도움 받을 때 사용
```

---

## 작물 분담표

각 담당이 받을 작물 (용량 균등 분배, 작물 단위로 통째로). **팀 회의에서 이름 채우기.**

| 담당 | 작물 (학습 그룹 + held-out*) | 용량 |
|---|---|---|
| **이우현** | 시설 05.상추, 시설 03.단호박, 시설 11.토마토 | ~231GB |
| **장순규** | 시설 02.고추, 시설 04.딸기, 시설 09.쥬키니호박, 노지 07.잎마름병토마토* | ~225GB |
| **문재원** | 노지 03.배추, 노지 01.고추, 시설 10.참외*, 노지 10.호박* | ~222GB |
| **박준환** | 노지 04.애호박, 노지 06.오이, 시설 01.가지* | ~202GB |
| **신현준** | 노지 08.콩, 노지 09.파, 시설 06.수박*, 노지 05.양배추* | ~260GB |

> `*` = held-out 그룹 (모델 학습 미사용, 개방형 일반화 평가 전용)
> 각 작물은 Training + Validation 모두 다운로드 필요

**원칙**:
- 한 작물은 반드시 한 사람이 통째로 받음 (Training + Validation)
- 디스크 여유가 부족하면 작물 1개씩 받고 전처리 후 원본 삭제 → 다음 작물 반복
- 자기 분담 확정되면 `preprocess.ipynb` 의 `MY_GROUPS` 에 입력

---

## 작업 흐름

```
[STEP 1] 다운로드 (자기 AI 도움받아서)
  └─ AI_PROMPT_TEMPLATE.md 의 「섹션 1 — 데이터 다운로드」를 자기 AI에게 통째로 줌
  └─ AI가 안내하는 대로 AIHubShell 설치/인증/다운로드
  └─ 결과를 표준 폴더 구조로 정리 (raw_data/.../Training/...zip)

[STEP 2] 전처리
  └─ config.py 에서 경로만 자기 환경에 맞게 수정
  └─ preprocess.ipynb 열고 MY_GROUPS 자기 담당 작물로 수정
  └─ 셀을 위에서부터 실행

[STEP 3] 결과 공유
  └─ preprocessed/*.zip 파일들을 팀 공유 폴더(또는 통합 담당자)에 전달

[STEP 4] 통합 (한 명만)
  └─ 모든 팀원 zip을 한 폴더에 모음
  └─ merge_manifests.ipynb 실행 → manifest_all.csv 생성 + 검증
```

---

## 사전 준비

### 1. Python 환경

Python 3.10 이상 권장.

```bash
pip install -r requirements.txt
# pip install pandas matplotlib jupyter tqdm
```

### 2. 폴더 구조 (자기 컴퓨터)

```
작업폴더/
├── preprocessing_code/          ← 이 폴더(github clone)
│
├── deeplearning_labeling.zip    ← AI Hub 라벨 (이미 보유)
│
├── raw_data/                    ← AI Hub에서 받은 원천 데이터
│   ├── facility/                  (시설)
│   │   ├── 02.고추/
│   │   │   ├── Training/
│   │   │   │   ├── 0.정상.zip
│   │   │   │   ├── 1.질병.zip
│   │   │   │   ├── 9.증강_(1).zip
│   │   │   │   ├── 9.증강_(2).zip
│   │   │   │   └── 9.증강_(3).zip
│   │   │   └── Validation/
│   │   │       ├── 0.정상.zip
│   │   │       ├── 1.질병.zip
│   │   │       └── 9.증강.zip
│   │   └── 03.단호박/...
│   │
│   └── outdoor/                   (노지)
│       └── 01.고추/...
│
└── preprocessed/                ← 전처리 결과 (자동 생성)
    ├── facility_02_고추/
    │   ├── images/
    │   └── manifest.csv
    └── facility_02_고추.zip
```

> **중요**: `raw_data/.../Training/` 안에 zip 파일을 그대로 두세요. **압축 해제 X**.

### 3. config.py 수정

`config.py` 상단의 경로만 자기 컴퓨터 경로에 맞게 수정:

```python
RAW_DATA_ROOT  = Path("./raw_data")              # 자기 경로로
LABEL_ZIP_PATH = Path("./deeplearning_labeling.zip")
OUTPUT_ROOT    = Path("./preprocessed")
```

**그 외 상수(SEED, 캡, 해상도 등)는 절대 수정 금지** — 팀원 결과가 달라집니다.

---

## AI Hub 다운로드 (자기 AI 도움받기)

AIHubShell은 OS별로 사용법이 다르고 인증·분할 압축 등 까다로운 부분이 있어, **자기 AI에게 도움받는 게 가장 빠릅니다.**

### 절차

1. **`AI_PROMPT_TEMPLATE.md` 열기**
2. **「🟢 섹션 1 — 데이터 다운로드 및 폴더 정리」 를 통째로 복사**
3. **자기 AI(ChatGPT/Claude/Gemini)에 붙여넣기**
4. **`[내 환경]` 부분의 OS, 셸, 담당 작물 채워서 전송**
5. **AI가 단계별로 안내** (설치 → 인증 → 다운로드 → 폴더 정리 → 검증)

### 필요한 파일키 목록

같은 폴더의 `filekey_table.md` 참조.
- 시설 데이터셋 키: `dataSetSn=153`
- 노지 데이터셋 키: `dataSetSn=147`
- 작물별 파일키 표 모두 정리됨

### 다운로드 결과는 반드시 이 구조로

```
raw_data/
├── facility/02.고추/
│   ├── Training/    *.zip 들
│   └── Validation/  *.zip 들
└── outdoor/01.고추/
    ├── Training/
    └── Validation/
```

- zip은 압축해제하지 말 것 (노트북이 zip 그대로 읽음)
- 폴더명/zip 파일명은 한국어 그대로 유지

---

## 실행: preprocess.ipynb (각자)

1. **Jupyter / VSCode / Colab** 에서 `preprocess.ipynb` 열기
2. 셀을 위에서부터 순서대로 실행
3. **"3. 자기 담당 그룹 선택"** 셀에서 `MY_GROUPS` 수정:
   ```python
   MY_GROUPS = [
       ('시설', '02.고추'),
       ('시설', '03.단호박'),
       ('노지', '09.파'),
   ]
   ```
4. 나머지 셀 그대로 실행 (마지막 "5. 실행" 셀이 본 처리)

**소요 시간**: 작물 1개당 약 30~90분 (CPU 코어, 디스크 속도에 따라).

각 셀이 진행 상황을 출력하므로 어디까지 됐는지 알 수 있습니다.

---

## 결과 전달

전처리가 끝나면 `preprocessed/` 폴더에:

```
preprocessed/
├── facility_02_고추.zip          ← 이걸 전달
├── facility_03_단호박.zip
├── outdoor_09_파.zip
├── facility_02_고추/             (참고용, 전달 안 해도 됨)
└── ...
```

`.zip` 파일들만 통합 담당자(또는 구글 드라이브 공유 폴더)에 전달.

각 zip 안에:
- `images/` 폴더 (전처리된 512×512 JPEG)
- `manifest.csv` (이미지별 메타데이터)

---

## 통합: merge_manifests.ipynb (한 명만)

모든 팀원 zip을 한 폴더에 모은 뒤:

1. **`merge_manifests.ipynb`** 열기
2. **"2. 입력 경로 지정"** 에서 `INPUT_DIR` 수정
3. 셀 순서대로 실행
4. 자동 검증:
   - 18개 그룹 모두 존재
   - split 비율 확인
   - 캡 5000 위반 여부
   - 원본 ID 누수 여부 (가장 중요)
   - 시각화 (그래프 4종)
5. `manifest_all.csv` 생성

---

## 자주 발생하는 문제

### "라벨 zip 없음"
→ `config.py` 의 `LABEL_ZIP_PATH` 경로 확인.

### "원본 zip 없음"
→ `raw_data/{facility|outdoor}/{작물폴더}/Training/` 안에 zip 있는지 확인.

### 처리 매우 느림
→ 작물 1개씩 처리 권장. `AI_PROMPT_TEMPLATE.md` 의 멀티프로세싱 추가 가이드 참고.

### 디스크 부족
→ 작물 1개 처리 → 원본 zip 삭제 → 다음 작물 다운로드 → 처리 반복.

### Jupyter 노트북에서 import 안 됨
→ 노트북과 같은 폴더에 .py 파일들이 있는지 확인.

---

## AI 도움 받기

각자 환경에 맞게 코드 수정 시 `AI_PROMPT_TEMPLATE.md` 참조. ChatGPT/Claude 등에 그대로 붙여넣어 도움을 받을 수 있습니다.

---

## 표준 설정 (절대 수정 금지)

| 항목 | 값 | 이유 |
|---|---|---|
| `RANDOM_SEED` | 42 | 팀원 분할 결과 일치 보장 |
| `TARGET_SIZE` | 512 | B0~B4 모두 지원 |
| `JPEG_QUALITY` | 90 | 품질과 용량의 절충 |
| `PADDING_RATIO` | 0.30 | 병변 주변 맥락 확보 |
| `MAX_PER_CLASS` | 5000 | 그룹×클래스 캡 (train만) |
| `SPLIT_RATIO` | (0.70, 0.15, 0.15) | 표준 분할 |

이 값들이 다르면 팀원 간 결과 불일치 발생 → 학습 데이터 망가짐.

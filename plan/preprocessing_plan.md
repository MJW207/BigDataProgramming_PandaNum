# 사전 전처리 작업 계획서

**작성일:** 2026-05-27
**작업 목표:** AI Hub 원본 데이터를 다운로드 → 전처리 → 구글 드라이브 저장 → Runpod 학습용 데이터셋 완성

---

## 1. 전체 작업 흐름

```
[Step 1] 다운로드 분담 (5명)
    ↓
[Step 2] 각자 로컬에서 AIHubShell로 다운로드
    ↓
[Step 3] 표준 전처리 코드로 각자 전처리 실행
    ↓
[Step 4] 그룹별 zip 묶기 → 구글 드라이브 업로드
    ↓
[Step 5] manifest.csv 통합 (한 명이 담당)
    ↓
[Step 6] Runpod 업로드 → 학습 시작
```

---

## 2. 다운로드 분담표 (5명 × 작물)

### 2.1 분담 원칙

- **각자 약 1.2TB / 5 ≈ 240GB 분담**
- 같은 작물은 한 사람이 통째로 받음 (Training + Validation 모두)
- 분할 시 학습/held-out 균등 분배
- 처리 후 약 4~5GB 결과물 업로드

### 2.2 분담안 (초안)

| 담당자 | 시설 작물 | 노지 작물 | 원본 용량 추정 |
|---|---|---|---|
| **A (팀장 이우현)** | 02.고추, 03.단호박 | 09.파 | ~180GB |
| **B (장순규)** | 04.딸기, 11.토마토 | 01.고추 | ~210GB |
| **C (문재원)** | 05.상추, 09.쥬키니호박 | 03.배추 | ~250GB |
| **D (박준환)** | 01.가지(held), 06.수박(held) | 04.애호박, 10.호박(held) | ~250GB |
| **E (신현준)** | 10.참외(held) | 06.오이, 08.콩, 05.양배추(held), 07.잎마름병토마토(held) | ~230GB |

> 정확한 분담은 팀 회의에서 조정. 각자 디스크 여유 공간 확인 후 결정.

### 2.3 학습 12그룹 + held-out 6그룹 분배 확인

**학습 12그룹:**
- 시설: 02.고추, 03.단호박, 04.딸기, 05.상추, 09.쥬키니호박, 11.토마토 (6)
- 노지: 01.고추, 03.배추, 04.애호박, 06.오이, 08.콩, 09.파 (6)

**held-out 6그룹:**
- 시설: 01.가지, 06.수박, 10.참외 (3)
- 노지: 05.양배추, 07.잎마름병(토마토), 10.호박 (3)

**완전 제외 4그룹 (다운로드 X):**
- 시설: 07.애호박, 08.오이, 12.포도
- 노지: 02.무

---

## 3. AIHubShell 다운로드 가이드

### 3.1 사전 준비

```bash
# AIHubShell 설치 (이미 받은 사람은 생략)
# https://www.aihub.or.kr 에서 AIHubShell 다운로드
# 사용자 인증 1회 진행
```

### 3.2 작물별 파일키 목록 (예시: 시설 02.고추)

작물별 받아야 할 모든 파일키는 `plan/filekey_table.md` 참조.

```bash
# 라벨링 데이터 (이미 deeplearning_labeling.zip에 있음 — 추가 다운 불필요)
# 원천 데이터만 받기

# 시설 02.고추 Training 원천
aihubshell -datasetkey 153 -filekey 46849  # 0.정상.zip
aihubshell -datasetkey 153 -filekey 46850  # 1.질병.zip
aihubshell -datasetkey 153 -filekey 46851  # 9.증강_(1).zip
aihubshell -datasetkey 153 -filekey 46852  # 9.증강_(2).zip
aihubshell -datasetkey 153 -filekey 46853  # 9.증강_(3).zip

# 시설 02.고추 Validation 원천
aihubshell -datasetkey 153 -filekey 46744
aihubshell -datasetkey 153 -filekey 46745
aihubshell -datasetkey 153 -filekey 46746
```

### 3.3 데이터셋 키

- **dataSetSn=153**: 시설 작물 질병 진단 (071)
- **dataSetSn=147**: 노지 작물 질병 진단 (073)

---

## 4. 전처리 코드 작성 계획

### 4.1 코드 위치 및 구조

```
BigDataProgramming_PandaNum/
├── preprocessing/
│   ├── preprocess.py              # 메인 전처리 스크립트
│   ├── crop_utils.py              # bbox 크롭 + 패딩 + letterbox 함수
│   ├── label_utils.py             # JSON 파싱 + 원본 ID 추출
│   ├── split_utils.py             # D-1 분할 (원본 ID 기준 70:15:15)
│   ├── sampling_utils.py          # 캡 5000 sampling
│   ├── config.py                  # 그룹/캡/해상도 등 상수
│   └── run_preprocess.sh          # 실행 셸 스크립트 (선택)
```

### 4.2 모듈별 책임

#### `config.py`
- 학습 12그룹 / held-out 6그룹 / 제외 4그룹 정의
- 환경별 crop_filter
- MAX_PER_CLASS = 5000
- 저장 해상도 = 512×512
- JPEG 품질 = 90
- 분할 비율 = 0.70 / 0.15 / 0.15
- 랜덤 시드 = 42

#### `label_utils.py`
- JSON 파싱: crop, disease, risk, points, original
- 원본 ID 추출:
  - 정상/질병 JSON: 파일명 그대로
  - 증강 JSON: `description.original` 필드 사용
- 메타데이터 정규화

#### `crop_utils.py`
- bbox 크롭 (points의 xtl, ytl, xbr, ybr)
- 30% 패딩 추가 (이미지 경계 클램핑)
- Letterbox resize 512×512 (cv2.INTER_LANCZOS4)
- bbox 결과가 너무 작으면 원본 이미지 사용

#### `split_utils.py`
- D-1 분할 구현:
  1. 그룹×클래스별 원본 ID 리스트 수집
  2. 원본 ID 단위로 train_test_split (stratified) 70:15:15
  3. 각 split의 원본 ID 집합 생성
  4. 모든 이미지(원본+증강)에 split 부여 (description.original 매칭)

#### `sampling_utils.py`
- 캡 5000 적용:
  - train만 적용, val/test 미적용
  - 그룹×클래스별로 random sampling
  - held-out은 캡 적용 안 함 (전체 평가)

#### `preprocess.py` (메인)
```
1. 입력 zip 경로 받기 (CLI 인자)
2. 라벨링 zip + 원천 zip 매칭
3. 그룹별 처리:
   a. 모든 JSON 파싱 → 메타 수집
   b. D-1 분할 (원본 ID 기준)
   c. 캡 5000 적용 (train만)
   d. 각 이미지에 대해:
      - 원본 이미지 로드 (zip에서 스트리밍)
      - bbox 크롭 + 패딩 + letterbox
      - 512×512 JPEG로 저장
      - manifest 항목 추가
4. 결과 zip 묶기 (그룹별)
5. manifest.csv 저장
```

### 4.3 출력 구조

```
preprocessed/
├── facility_02_고추.zip               ← 작물별 zip
│   ├── images/
│   │   ├── img_000001.jpg            (512×512)
│   │   ├── img_000002.jpg
│   │   └── ...
│   └── manifest.csv                   (이 작물의 manifest)
├── facility_03_단호박.zip
├── ...
├── outdoor_09_파.zip
└── README.txt                        ← 처리 정보 (날짜, 버전 등)
```

### 4.4 manifest.csv 형식

| 컬럼 | 설명 | 예시 |
|---|---|---|
| file | 이미지 파일명 | img_000001.jpg |
| env | 환경 | facility |
| crop_folder | 작물 폴더명 | 02.고추 |
| crop_code | crop 코드 | 2 |
| disease | disease 코드 | 3 |
| risk | 진행단계 (0~3) | 1 |
| grow | 생육 단계 | 13 |
| original_id | 원본 ID (증강은 원본 매칭) | V006_77_..._10016 |
| is_aug | 증강 여부 | False / True |
| split | train/val/test/heldout | train |
| group_type | 그룹 분류 | train_group / heldout_group |

### 4.5 표준화 — 모두 같은 결과 보장

```python
# config.py 에 명시
RANDOM_SEED = 42
TARGET_SIZE = 512
JPEG_QUALITY = 90
PADDING_RATIO = 0.3
SPLIT_RATIO = (0.70, 0.15, 0.15)
MAX_PER_CLASS = 5000
INTERPOLATION = cv2.INTER_LANCZOS4
```

→ 모든 팀원이 같은 코드 + 같은 seed로 실행하면 결과 동일

### 4.6 멀티프로세싱

```python
from multiprocessing import Pool
# 작물 그룹별 또는 이미지 단위로 병렬 처리
# 4~8 코어 활용으로 처리 시간 단축 (7~10시간 → 1~2시간)
```

---

## 5. manifest 통합 방법

### 5.1 통합 워크플로우

```
[각 팀원 결과]
A: facility_02_고추.zip, facility_03_단호박.zip, outdoor_09_파.zip
   + 각 zip 안에 manifest.csv

[통합 담당자 작업]
1. 모든 zip을 한 폴더에 모음
2. 통합 스크립트 (merge_manifests.py) 실행
3. 작물별 manifest.csv를 하나로 합침
4. 전체 manifest_all.csv 생성
5. 검증: 그룹 수, 클래스 분포, split 비율 확인
6. 최종 결과 → Runpod 업로드
```

### 5.2 통합 스크립트 (`merge_manifests.py`)

```
입력: 18개 작물 zip (또는 압축 해제된 폴더)
처리:
  1. 각 zip 안의 manifest.csv 읽음
  2. 컬럼 일치 확인
  3. 전체 manifest_all.csv 생성
  4. 통계 출력:
     - 그룹별 총량
     - 클래스별 분포
     - split 비율 (train/val/test/heldout)
     - 누락 그룹 체크
  5. 검증 실패 시 어느 작물에 문제가 있는지 알림
출력: manifest_all.csv
```

### 5.3 검증 체크리스트

- [ ] 18개 그룹 모두 존재 (12 학습 + 6 held-out)
- [ ] 각 그룹×클래스 모두 존재 (시설 애호박 말기 제외)
- [ ] split 비율: train 60~65% / val 18~20% / test 18~20%
- [ ] 원본 ID 단위로 split 분리 확인 (같은 ID가 여러 split에 없는지)
- [ ] 캡 5000 적용 확인 (train에서 max 5000장)
- [ ] 이미지 파일이 manifest에 모두 존재
- [ ] 이미지 해상도 512×512 일치

---

## 6. Runpod 업로드 및 배포

### 6.1 업로드 구조

```
Google Drive / 또는 Runpod Storage
└── preprocessed_v1/
    ├── manifest_all.csv               (통합 manifest)
    ├── facility_02_고추.zip
    ├── facility_03_단호박.zip
    ├── ... (18개 작물 zip)
    └── README.md                       (버전, 처리 일자, 통계)
```

### 6.2 Runpod에서 사용

학습 시작 전:
```bash
# 1. 데이터 다운로드 (Google Drive에서)
# 2. 모든 zip 압축 해제
# 3. manifest_all.csv 로드
# 4. dataset.py가 manifest 기준으로 이미지 로드
```

---

## 7. 작업 일정 (제안)

| 단계 | 작업 | 담당 | 기간 |
|---|---|---|---|
| 1 | 전처리 코드 작성 + 테스트 | 문재원 (메인) | 2026-05-27 ~ 05-29 |
| 2 | 1명이 1개 작물로 코드 검증 | 한 명 | 2026-05-30 |
| 3 | 분담표 확정 + AIHubShell 설치 확인 | 전원 | 2026-05-30 |
| 4 | 각자 다운로드 + 전처리 | 전원 (병렬) | 2026-05-31 ~ 06-02 |
| 5 | 결과 zip 업로드 + manifest 통합 | 통합 담당 | 2026-06-03 |
| 6 | Runpod 업로드 + 학습 시작 | 전원 | 2026-06-04 ~ |

**버퍼**: 일정 1~2일 여유. 학습 시작 후 발표(06-10)까지 5~6일.

---

## 8. 위험 요소 및 대응

| 위험 | 대응 |
|---|---|
| 일부 팀원 AIHubShell 다운로드 속도 느림 | 다른 팀원이 추가 분담 (분담표 재조정) |
| 디스크 용량 부족 | 작물 1개씩 받고 즉시 전처리 후 원본 삭제 |
| 전처리 코드 버그로 결과 불일치 | 1명이 코드 검증 후 배포, 같은 seed 사용 |
| 구글 드라이브 업로드 느림 | 작물별 zip이 충분히 큼 (Drive는 큰 파일에 유리) |
| AI Hub 다운로드 권한 만료 | 다운로드 권한 사전 확인 (다같이) |

---

## 9. 다음 작업 (이 계획서 승인 후)

1. `preprocessing/config.py` 작성
2. `preprocessing/label_utils.py` 작성 (JSON 파싱 + 원본 ID 추출)
3. `preprocessing/crop_utils.py` 작성 (bbox 크롭 + letterbox)
4. `preprocessing/split_utils.py` 작성 (D-1 분할)
5. `preprocessing/sampling_utils.py` 작성 (캡 5000)
6. `preprocessing/preprocess.py` 작성 (메인 통합)
7. 한 작물(예: 노지 콩)로 동작 검증
8. 전체 팀에 배포

# AI에게 도움 받기 — 프롬프트 템플릿

자기 환경에 맞게 데이터 다운로드/전처리 코드를 수정할 때 AI(ChatGPT/Claude/Gemini 등)에 활용하는 템플릿입니다. **아래 내용을 복사해서 AI에게 통째로 붙여넣으면 됩니다.**

> **사용 흐름**
> 1. 「**섹션 1 — 데이터 다운로드**」 프롬프트를 AI에게 줘서 자기 환경에 맞게 다운로드 받고 정리하기
> 2. 그 결과 폴더 위치를 `config.py` 에 설정
> 3. `preprocess.ipynb` 실행하면 노트북이 표준 구조를 읽음
> 4. 처리 중 문제 생기면 「**섹션 2 — 전처리 문제 해결**」 프롬프트 활용

---

## 🟢 섹션 1 — 데이터 다운로드 및 폴더 정리 (가장 먼저!)

**자기 AI(ChatGPT/Claude 등)에게 이걸 통째로 붙여넣고 도움 받으세요.**

```
나는 한성대학교 빅데이터프로그래밍 팀 프로젝트에서 AI Hub의 작물 질병 이미지를 
다운받아 사전 전처리하는 작업을 맡았어. 팀원들과 같은 전처리 노트북을 공유해 쓰는데,
이 노트북은 다음과 같은 표준 폴더 구조를 기대해.

[표준 폴더 구조 — 노트북이 읽는 위치]

raw_data/                                       ← 다운받은 원본을 여기로 정리
├── facility/                                     (시설 작물)
│   ├── 02.고추/
│   │   ├── Training/
│   │   │   ├── 0.정상.zip
│   │   │   ├── 1.질병.zip
│   │   │   └── 9.증강_(1).zip, 9.증강_(2).zip, 9.증강_(3).zip
│   │   └── Validation/
│   │       ├── 0.정상.zip
│   │       ├── 1.질병.zip
│   │       └── 9.증강.zip
│   ├── 03.단호박/
│   │   ├── Training/...
│   │   └── Validation/...
│   └── ...
└── outdoor/                                      (노지 작물)
    ├── 01.고추/
    │   ├── Training/...
    │   └── Validation/...
    └── ...

[중요한 규칙]
- 받은 zip 파일은 압축해제하지 말고 zip 그대로 둘 것 (노트북이 스트리밍으로 읽음)
- 폴더명은 정확히 한국어로: facility / outdoor, Training / Validation, 0.정상.zip 등
- 작물 폴더 이름은 "02.고추" 처럼 번호 점 이름 형태 유지

[다운로드 방법]
- AI Hub (https://aihub.or.kr) 에서 AIHubShell 도구 사용
- 데이터셋 키:
  · 시설 작물 질병 진단 (071): dataSetSn = 153
  · 노지 작물 질병 진단 (073): dataSetSn = 147
- 작물별 파일키는 별도 표(filekey_table.md)에 있음

[내 환경]
- OS: [Windows / Mac / Linux 중 선택]
- 셸: [PowerShell / Git Bash / Mac Terminal / WSL 등]
- AIHubShell 설치 여부: [예 / 아니오, 처음]
- 디스크 여유: [예: 300GB]
- 내가 담당한 작물: [예: 시설 02.고추, 시설 03.단호박, 노지 09.파]

[도움 요청]
다음을 단계별로 알려줘:
1. AIHubShell 설치 및 사용자 인증 방법 (내 OS 기준)
2. 내가 담당한 작물의 모든 파일을 받는 명령어 (또는 자동화 스크립트)
3. 다운받은 파일이 .zip.part0, .part1 식으로 분할되어 있으면 합치는 방법
4. 받은 zip을 위의 [표준 폴더 구조]대로 정리하는 방법 (셸 명령어 또는 스크립트)
5. 정리 끝났을 때 폴더 구조가 맞는지 검증하는 방법
```

---

## 🔵 섹션 2 — 전처리 문제 해결 (다운로드 완료 후)

## 📋 AI에게 줄 컨텍스트 (이 섹션을 통째로 붙여넣기)

```
나는 한성대학교 빅데이터프로그래밍 팀 프로젝트(판다넘)에서 
"환경 통합 작물 질병 진행단계 분류" 모델을 만들고 있어.

내가 맡은 일은:
- AI Hub의 작물 질병 이미지 데이터를 다운받고
- 표준화된 전처리 코드로 가공한 뒤
- 결과를 팀에 공유하는 거야.

전처리 흐름:
1. JSON 라벨(deeplearning_labeling.zip)에서 bbox/risk/disease/original 추출
2. 원본 이미지(3024x3024)를 bbox 크롭 + 30% 패딩
3. Letterbox resize로 512x512 JPEG 저장 (cv2.INTER_LANCZOS4)
4. 원본 ID 기준 70:15:15 분할 (같은 원본의 증강은 같은 split)
5. train만 그룹×클래스별 MAX_PER_CLASS=5000 캡 적용

표준 설정 (절대 수정 금지):
- RANDOM_SEED = 42
- TARGET_SIZE = 512
- JPEG_QUALITY = 90
- PADDING_RATIO = 0.30
- MAX_PER_CLASS = 5000
- SPLIT_RATIO = (0.70, 0.15, 0.15)

코드 파일:
- preprocess.ipynb:     메인 실행 노트북 (각 팀원이 사용, 단계별 셀)
- merge_manifests.ipynb: 통합 노트북 (한 명만, 검증 시각화 포함)
- config.py:            경로 + 그룹 정의 + 표준 설정
- label_utils.py:       JSON 파싱, 원본 ID 추출
- crop_utils.py:        bbox 크롭 + Letterbox resize
- split_utils.py:       D-1 원본 ID 분할
- sampling_utils.py:    캡 5000 적용

내 환경:
- OS: [Windows/Mac/Linux 중 하나 적기]
- Python: [3.10 / 3.11 / 3.12 중 하나]
- CPU 코어: [예: 8코어]
- RAM: [예: 16GB]
- 디스크 여유: [예: 500GB]
- 담당 작물: [예: 시설 02.고추, 시설 03.단호박, 노지 09.파]

다음 문제를 도와줘:

[여기에 내 문제/요청 적기]
```

---

## 🔹 자주 묻는 질문 (FAQ)

### Q1. "내 컴퓨터 OS에 맞게 경로 설정 도와줘"

```
내 OS는 [Windows / Mac / Linux]야.
현재 config.py의 경로:
  RAW_DATA_ROOT = Path("./raw_data")
  LABEL_ZIP_PATH = Path("./deeplearning_labeling.zip")
  OUTPUT_ROOT = Path("./preprocessed")

내 실제 폴더 위치는:
  - raw_data 폴더: [예: D:\AIhub\raw_data]
  - 라벨 zip: [예: D:\AIhub\deeplearning_labeling.zip]
  - 결과 저장: [예: D:\AIhub\preprocessed]

config.py를 어떻게 수정해야 할지 알려줘.
Windows 경로의 백슬래시 처리도 신경써줘.
```

### Q2. "처리가 너무 느려. 멀티프로세싱 추가해줘"

```
preprocess.ipynb 의 process_group() 함수가 zip 안의 이미지를 하나씩 처리하는데, 
내 CPU는 [N]코어라서 병렬 처리하고 싶어.

조건:
- multiprocessing.Pool 사용
- 워커 수는 CPU 코어 수의 절반 (예: 8코어면 4 워커)
- 메모리 부담 줄이기 위해 imap 사용
- tqdm으로 진행률 표시
- 같은 결과 보장 (랜덤성 없는 부분이라 멀티프로세싱 안전함)

노트북 내의 process_group() 셀을 멀티프로세싱 버전으로 다시 작성해줘.
다른 모듈(config.py, crop_utils.py 등)은 건드리지 말고.
주의: Windows에서 multiprocessing은 if __name__ == '__main__' 가드 필요할 수 있음.
```

### Q3. "AI Hub에서 받은 파일이 분할되어 있는데 합쳐야 해"

```
AIHubShell로 받은 파일이 .zip.part0, .zip.part1 같이 분할되어 있어.
내 OS는 [Windows/Mac/Linux]야.

다음 zip이 분할되어 있어:
  - [예: 9.증강_(1).zip.part0, .part1, .part2]

이걸 하나의 zip으로 합치는 명령어 알려줘.
(파일이 손상되지 않게 안전하게)
```

### Q4. "디스크 용량이 부족해. 한 작물씩 처리하고 원본 삭제하는 워크플로우 만들어줘"

```
내 디스크 여유는 [N]GB뿐이고, 작물 한 개당 원본 데이터가 50~110GB라 한꺼번에 다 받을 수 없어.

다음 워크플로우를 자동화하는 스크립트(또는 셸 스크립트) 만들어줘:

1. 작물 A 다운로드 (AIHubShell)
2. preprocess.ipynb 실행 (또는 jupyter nbconvert --execute) (작물 A)
3. 원본 zip 삭제 (작물 A)
4. 작물 B 다운로드
5. preprocess.ipynb 실행 (작물 B)
6. ... 반복

내 담당 작물 목록:
  - [예: 시설 02.고추 (파일키 46849, 46850, 46851, 46852, 46853, 46744, 46745, 46746)]
  - [예: 시설 03.단호박 (...)]
  - ...

OS는 [Windows PowerShell / Mac bash / Linux bash]야.
```

### Q5. "특정 작물만 다시 처리해줘"

```
이미 전처리한 결과 중 [예: 시설 02.고추] 만 다시 처리하고 싶어.

이유: [예: 처음 처리할 때 라벨 zip을 잘못 지정했음]

preprocess.ipynb 의 MY_GROUPS 를 해당 작물만 넣고 다시 돌리면 되는지,
이전 결과 폴더는 자동 덮어쓰기 되는지 알려줘.
```

### Q6. "메모리 부족 (OOM) 에러가 발생해"

```
preprocess.ipynb 실행 중 다음 에러 발생:
[에러 메시지 그대로 붙여넣기]

내 RAM은 [N]GB야.

원인 분석하고 메모리 부담 줄이는 방향으로 코드 수정해줘.
가능한 방법:
- 이미지를 한 번에 하나씩만 처리
- zip을 chunk 단위로 읽기
- garbage collection 명시적 호출
```

### Q7. "처리 결과가 다른 팀원과 다르게 나와"

```
같은 작물을 처리했는데 결과 manifest의 split 분포가 팀원과 달라.

확인하고 싶은 것:
1. RANDOM_SEED, MAX_PER_CLASS 등 config.py 표준 값이 그대로인지
2. label_utils.py의 원본 ID 추출 로직이 변경되지 않았는지
3. sampling_utils.py의 random 사용 부분에서 seed가 제대로 고정되었는지

내가 처리한 작물: [예: 시설 02.고추]
내 manifest의 train 분포: [예: r0:5000, r1:5000, r2:5000, r3:3817]
팀원 manifest의 train 분포: [예: r0:5000, r1:4900, r2:5000, r3:3800]

차이의 원인을 분석해줘.
```

### Q8. "내 환경에서 OpenCV 설치 오류"

```
pip install opencv-python-headless 실행 시 다음 에러:
[에러 메시지]

내 환경:
- OS: [...]
- Python: [...]
- pip 버전: [...]

대안 설치 방법 알려줘.
```

---

## 🔹 코드 수정 시 주의사항 (AI에게 알려줄 것)

```
내가 수정 가능한 것:
- config.py 의 경로 (RAW_DATA_ROOT, LABEL_ZIP_PATH, OUTPUT_ROOT)
- preprocess.py 의 실행 인자
- 멀티프로세싱/병렬 처리 (성능 개선)
- 로깅/진행률 표시 개선
- 에러 핸들링 추가

내가 수정 금지인 것 (팀원과 결과 불일치 발생):
- config.py 의 표준 값 (SEED, SIZE, CAP, RATIO 등)
- label_utils.py 의 원본 ID 추출 로직
- crop_utils.py 의 letterbox/bbox 로직
- split_utils.py 의 D-1 분할 로직
- sampling_utils.py 의 캡 적용 로직
- manifest.csv 의 컬럼 구조

수정해주는 코드가 위 "금지" 항목에 해당하지 않는지 먼저 확인해줘.
```

---

## 🔹 검증 (전처리 결과 점검)

처리 끝나면 다음을 AI에게 요청:

```
preprocessed/{내가_처리한_그룹}/manifest.csv 의 첫 5줄과 통계 확인해줘.

[manifest.csv 첫 5줄 붙여넣기]

확인할 것:
1. 컬럼이 올바른가 (file, env, crop_folder, crop_code, disease, risk, grow, original_id, is_aug, split, group_type, group_id)
2. split 비율이 train 60~65% / val 18~20% / test 18~20% 인가 (학습 그룹의 경우)
3. risk 컬럼이 0~3 정수만 있는가
4. 같은 original_id가 여러 split에 있지 않은가 (누수 검사)
```

---

## 🔹 마무리 시 AI에게 요청

```
처리가 다 끝났어. 결과 zip을 팀에 공유하기 전에 다음을 확인해줘:

1. preprocessed/ 폴더의 모든 작물 zip 파일 목록
2. 각 zip의 용량 (대략 1~5GB 예상)
3. zip 안에 manifest.csv가 잘 포함되어 있는지 (zip -sf 같은 명령으로)
4. 잘못 들어간 파일은 없는지 (테스트 파일, 캐시 등)
```

---

## 🔹 자주 사용하는 ChatGPT/Claude 명령

```
이 코드를 [Windows 환경] 에 맞게 수정해줘. 표준 값(SEED, CAP 등)은 건드리지 말고 경로만 수정해.
```

```
이 에러 메시지를 분석해서 해결책 알려줘:
[에러 메시지]
```

```
이 코드에 tqdm 진행률 표시랑 멀티프로세싱 추가해줘. 결과는 같아야 해.
```

```
이 코드를 한 작물씩 자동으로 다운로드 → 전처리 → 원본 삭제하는 워크플로우로 만들어줘.
```

---

이 템플릿을 자기 상황에 맞게 채워서 AI에게 물어보면 정확한 답을 받을 수 있습니다.

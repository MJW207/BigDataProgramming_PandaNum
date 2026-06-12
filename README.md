# 환경 통합 작물 질병 진행단계 분류 모델

> 빅데이터 프로그래밍 팀 프로젝트 — **팀 판다넘**  
> EfficientNet-B3 + CBAM | 시설·노지 통합 학습 | 개방형 일반화 검증

---

## 제출 저장소 안내

본 저장소는 빅데이터프로그래밍 프로젝트의 코드 제출용 저장소입니다.

- 데이터 전처리 코드: `plan/preprocessing_code/`
- 모델 구현 코드: `model.py`, `dataset.py`
- 최종 학습 및 평가 노트북: `experiments/final/crop_disease_notebook_num6_last.ipynb`
- 반복 실험 노트북 및 하이퍼파라미터 기록: `experiments/runs/`
- 프로젝트 설계 및 피드백 문서: `docs/`

원본 AI Hub 이미지, 학습된 모델 가중치, 대용량 영상 파일은 용량 문제로 저장소에 포함하지 않습니다. 데이터는 AI Hub에서 별도로 내려받은 뒤 README의 실행 순서에 따라 전처리하여 사용합니다.

---

## 프로젝트 개요

작물 질병의 **병명**이 아닌 **진행 단계(정상 → 초기 → 중기 → 말기)** 를 분류하는 모델.  
시설과 노지를 통합 학습하여 환경 편향 없이 동작하고, 학습에 없던 새로운 작물에서도 단계 추정 가능 여부를 직접 검증한다.

### 연구 질문

> *시설과 노지를 통합한 단일 모델이, 환경에 치우치지 않고, 나아가 학습에 없던 작물에서도 질병 진행 단계를 분류할 수 있는가?*

### 가설

| 가설 | 검증 결과 |
|---|---|
| 가설 1: 통합 모델이 두 환경에서 안정적으로 동작한다 | 시설 F1=0.759 / 노지 F1=0.726 — **차이 0.033으로 지지** |
| 가설 2: 학습 외 새 작물에서도 정상과 발병을 구분할 수 있다 | held-out 정상 F1=**0.941** — **지지** |

---

## 주요 결과

| 실험 | Val Macro F1 | Test Macro F1 |
|---|---|---|
| Exp-1: EfficientNet-B0 (BL-2 Fine-tuning) | 0.7545 | — |
| **Exp-2: EfficientNet-B3 (핵심 모델)** | **0.7628** | **0.7447** |
| Exp-4: B3 + Soft Labeling | 0.7661 | — |

### 환경별 분리 평가 (Exp-2 B3)

| 환경 | Macro F1 | 정상 | 초기 | 중기 | 말기 |
|---|---|---|---|---|---|
| 시설 | 0.7590 | 0.994 | 0.722 | 0.590 | 0.730 |
| 노지 | 0.7262 | 0.991 | 0.731 | 0.588 | 0.595 |
| 통합 | 0.7447 | 0.993 | 0.726 | 0.589 | 0.671 |

### 개방형 일반화 평가 (held-out 5작물)

| 작물 | 환경 | F1 | 샘플 수 |
|---|---|---|---|
| 양배추 | 노지 | 0.5674 | 36,078 |
| 잎마름병(토마토) | 노지 | 0.5110 | 15,070 |
| 호박 | 노지 | 0.5540 | 17,520 |
| 수박 | 시설 | 0.4570 | 27,576 |
| 참외 | 시설 | 0.5024 | 28,690 |
| **종합** | — | **0.5478** | **124,934** |

> 일반화 유지율: 0.5478 / 0.7628 = **71.8%**  
> 새 작물에서도 **정상 F1 = 0.941** — 정상/발병 이진 구분 능력은 유지됨

---

## 데이터셋

- **출처**: AI Hub — 시설 작물 질병 진단(071) / 노지 작물 질병 진단(073)
- **전체 규모**: 약 1TB, **388,299장** (캡 적용 전, 15작물 전체)
- **클래스**: 정상(0) / 초기(1) / 중기(2) / 말기(3) — JSON `risk` 항목 기준

### 작물 구성

| 구분 | 환경 | 작물 |
|---|---|---|
| 학습 (10그룹) | 시설 | 단호박, 딸기, 상추, 쥬키니호박, 토마토 |
| 학습 (10그룹) | 노지 | 배추, 애호박, 오이, 콩, 파 |
| Held-out (5그룹) | 시설 | 수박, 참외 |
| Held-out (5그룹) | 노지 | 양배추, 잎마름병(토마토), 호박 |
| 제외 | 시설 | 애호박(말기 0장), 오이·포도(샘플 부족) |
| 제외 | 노지 | 무(샘플 부족) |

### 분할 전략 (D-1 Split)

- 원본 `original_id` 기준 **70:15:15** stratified 분할
- 같은 원본의 증강 이미지가 train/val/test에 분산되는 **데이터 누수 방지**
- 학습 그룹: 캡 적용(MAX_PER_CLASS=5,000) → train split만
- Held-out 그룹: 분할 없이 전체 평가용

---

## 모델 아키텍처

```
입력 이미지 (300×300×3)
    │
    ▼
EfficientNet-B3 features   ← ImageNet 사전학습, 2단계 fine-tuning
    │
    ▼
CBAM (Channel + Spatial Attention)
    │
    ▼
Global Average Pooling → 1536-d
    │
    ▼
Dropout(0.5) → Linear(1536→512) → BN → ReLU → Dropout(0.4)
    │
    ▼
Linear(512→4) → [정상, 초기, 중기, 말기]
```

- **CBAM 위치**: EfficientNet features 전체 이후, GAP 이전
- **B3 파라미터**: 약 12M (B0=5.3M 대비 2.3배)
- **학습 시간**: B3 ≈ B0 × 2.4배

---

## 학습 전략

### 2단계 Fine-tuning

| 단계 | 에폭 | 동결 | LR |
|---|---|---|---|
| Phase 1 | 1~7 | features 동결 (CBAM + fc만 학습) | 2.5e-3 |
| Phase 2 | 8~35 | 전체 unfreeze + 차등 lr | features[:4]×0.1, features[4:]×0.2, CBAM×0.5, fc×1.0 |

### 클래스 불균형 처리

| 방법 | 내용 |
|---|---|
| 데이터 캡 | train split 그룹×클래스당 최대 5,000장 |
| WeightedRandomSampler | weight = 1 / count(group_id, risk) — 에폭당 60,000샘플 |
| FocalLoss | gamma=3.0, weight=None (이중보정 방지) |

불균형 비율: 캡 적용 전 2.4:1 → 적용 후 **1.3:1**

### 하이퍼파라미터 (Run#7 최종)

```python
LR            = 5e-4
WEIGHT_DECAY  = 5e-4
DROPOUT       = 0.5
FREEZE_EPOCHS = 7
EPOCHS        = 35
PATIENCE      = 7
BATCH_SIZE    = 64
NUM_SAMPLES_EP= 60_000
GAMVAL        = 3.0    # FocalLoss gamma
```

### Soft Labeling (Exp-4)

인접 단계에 확률 α/2씩 분산 — 질병 진행의 연속성 반영

```python
alpha = 0.10
# 중기(2) 실제 라벨: [0.00, 0.05, 0.90, 0.05]
# 정상(0) 실제 라벨: [0.947, 0.053, 0.00, 0.00]  ← 경계는 normalize
```

### TTA (Test-Time Augmentation)

Held-out 평가 시 3뷰(원본 / 좌우반전 / 상하반전) softmax 평균

---

## 전처리 파이프라인

```
원본 이미지 (3024×3024 등)
    │
    ▼ JSON bbox 기준 크롭 + 30% 패딩
    │
    ▼ Letterbox resize 512×512
    │   - 종횡비 유지, 회색 패딩 (114, 114, 114)
    │   - INTER_LANCZOS4 (고품질 다운샘플링)
    │   - JPEG quality=90 저장
    │
    ▼ 학습 시 (B3 기준)
    │   Resize(332) → RandomCrop(300) → Flip / Rotation / Grayscale
    │
    ▼ 추론 시
        Resize(300×300) → Normalize
```

정규화 통계 (실제 데이터 계산):
- mean = [0.444, 0.490, 0.364]
- std  = [0.229, 0.224, 0.225]  (ImageNet과 유사)

---

## 파일 구조

```
BigDataProgramming_PandaNum/
│
├── dataset.py                  # 데이터셋 클래스, transform, WeightedSampler
├── model.py                    # EfficientNetCropDisease, CBAM, SoftLabelLoss
├── plot_linear.py              # BL-1 학습 곡선 시각화
├── plot_results.py             # Fine-tuning 학습 곡선 시각화
├── crop_disease_notebook.ipynb # 초기 실험 노트북
├── extended_eda_result.csv     # EDA 결과 (15작물 × 클래스별 수량)
│
├── plan/
│   ├── preprocessing_code/     # 전처리 모듈
│   │   ├── config.py           # 공통 설정 (경로, 상수, 그룹 정의)
│   │   ├── crop_utils.py       # bbox 크롭 + Letterbox resize
│   │   ├── label_utils.py      # JSON 라벨 파싱
│   │   ├── sampling_utils.py   # 캡 적용 샘플링
│   │   ├── split_utils.py      # D-1 분할 (원본 ID 기준)
│   │   ├── preprocess.ipynb    # 전처리 실행 노트북
│   │   └── merge_manifests.ipynb
│   └── training_code/
│       ├── dataset.py          # (plan 단계 버전)
│       └── model.py
│
├── experiments/
│   ├── runs/                   # 실험 반복 노트북 (num1~6)
│   └── final/
│       └── crop_disease_notebook_num6_last.ipynb  # 최종 학습 노트북
│
└── 데이터확인/
    ├── label_eda.ipynb         # 라벨 EDA
    ├── analyze.py              # 분포 분석
    ├── check_split.py          # 분할 검증
    ├── d1_analysis.py          # D-1 분할 분석
    └── d1_per_crop.py          # 작물별 분포
```

---

## 환경 설정

```bash
pip install -r requirements.txt
```

GPU 환경 (RunPod 기준):
- GPU: NVIDIA RTX PRO 4500 Blackwell
- CUDA: AMP(Mixed Precision) 활성화
- num_workers=8, prefetch_factor=4

주요 라이브러리:

- PyTorch / torchvision
- timm
- OpenCV
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- tqdm, Pillow

---

## 실행 순서

```
1. 전처리
   plan/preprocessing_code/preprocess.ipynb
   → 원본 이미지 bbox 크롭 + Letterbox 512×512 + manifest_all.csv 생성

2. 학습
   experiments/final/crop_disease_notebook_num6_last.ipynb
   → Exp-1(B0) → Exp-2(B3) → Exp-4(B3+SoftLabel) 순서로 실행

3. 평가
   동일 노트북 내 시설/노지 분리 평가, held-out 평가, Grad-CAM 시각화
```

### 재현 시 주의사항

- AI Hub 원본 데이터는 저장소에 포함되어 있지 않으므로 별도 다운로드가 필요합니다.
- `config.py` 및 노트북 내부의 데이터 경로는 실행 환경에 맞게 수정해야 합니다.
- train split에만 `MAX_PER_CLASS=5,000` cap이 적용되며, validation/test/held-out 평가는 원 분포 기준으로 수행합니다.
- WeightedRandomSampler는 학습 DataLoader에만 적용하고, validation/test/held-out 평가에는 적용하지 않습니다.
- 최종 코드 기준 FocalLoss gamma는 `3.0`입니다.

---

## 실험 이력

| 노트북 | 주요 변경 내용 |
|---|---|
| num1 | B0 베이스라인 구축 |
| num2 | WeightedSampler 도입, 그룹 키 설계 |
| num3 | FocalLoss 적용, 데이터 캡 실험 |
| num4 | B3 도입, 2단계 fine-tuning 설계 |
| num5 | 차등 lr, CosineAnnealingWarmRestarts |
| num6 | NUM_SAMPLES_EP 40k→60k, gamma 2.0→3.0 |
| **num6_last** | **최종 설정 확정, Soft Labeling, TTA, held-out 평가** |

---

## 한계 및 향후 과제

- 초기·중기 혼동이 가장 큰 오류 원인 — 진행 단계 경계의 본질적 모호성
- 학습 외 작물의 중기 F1=0.381, 초기 F1=0.321 — 세밀한 단계 구분은 제한적
- 라벨 품질 불일치 가능성 (AI Hub 어노테이션 기준 불명확)
- 실시간 추론 최적화 미수행 (스마트팜 배포 시 필요)

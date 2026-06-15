# 환경 통합 작물 질병 진행단계 분류 모델

빅데이터프로그래밍 팀 프로젝트 **판다넘(PandaNum)** 최종 코드 제출 저장소입니다.

AI Hub 시설·노지 작물 질병 이미지 데이터를 활용하여 작물 질병의 진행단계를 **정상 / 초기 / 중기 / 말기** 4개 클래스로 분류합니다. 최종 모델은 **EfficientNet-B3 + CBAM** 구조이며, 시설·노지 통합 학습과 held-out 평가를 통해 환경 일반화 가능성을 확인했습니다.

---

## 주요 결과

| 평가 항목 | Macro F1 |
|---|---:|
| 통합 Test | 0.7447 |
| 시설 Test | 0.7590 |
| 노지 Test | 0.7262 |
| held-out 5그룹 | 0.5478 |

- 시설·노지 Test 성능 차이: **0.0328**
- held-out 일반화 유지율: **71.8%**
- held-out 정상 F1: **0.941**
- EfficientNet-B0 fine-tuning Val Macro F1: **0.7545**
- EfficientNet-B3 Val Macro F1: **0.7628**
- Soft Labeling 적용 Val Macro F1: **0.7661**

---

## 저장소 구조

```text
BigDataProgramming_PandaNum/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   └── model.py
├── preprocessing/
│   ├── config.py
│   ├── crop_utils.py
│   ├── label_utils.py
│   ├── sampling_utils.py
│   └── split_utils.py
└── notebook/
    └── crop_disease_notebook_num6_last.ipynb
```

---

## 주요 파일

| 경로 | 설명 |
|---|---|
| `src/model.py` | EfficientNet-B3 + CBAM 모델 정의 |
| `src/dataset.py` | 데이터셋 로더, transform, WeightedRandomSampler 구성 |
| `preprocessing/config.py` | 데이터 경로, 학습/held-out 그룹, 클래스 설정 |
| `preprocessing/crop_utils.py` | bbox crop, padding, letterbox resize 유틸리티 |
| `preprocessing/label_utils.py` | AI Hub JSON 라벨 파싱 |
| `preprocessing/sampling_utils.py` | train split cap 적용 |
| `preprocessing/split_utils.py` | `original_id` 기준 stratified split |
| `notebook/crop_disease_notebook_num6_last.ipynb` | 최종 학습, 평가, 분석 노트북 |

---

## 데이터

본 저장소에는 원본 이미지 데이터와 학습된 모델 가중치를 포함하지 않습니다.

사용 데이터:

- AI Hub 노지 작물 병해충 진단 데이터
- AI Hub 시설 작물 병해충 진단 데이터

데이터 구성:

- 전체 데이터: 388,299장
- 학습 그룹: 10개 작물·환경 그룹
- held-out 평가 그룹: 5개 작물·환경 그룹
- 클래스: 정상, 초기, 중기, 말기

전처리 요약:

- JSON `risk` 값을 진행단계 라벨로 사용
- bbox 기준 crop
- 30% padding
- 512x512 letterbox resize
- `original_id` 기준 70:15:15 stratified split
- train split에만 `MAX_PER_CLASS=5,000` cap 적용

---

## 환경 설정

```bash
pip install -r requirements.txt
```

주요 라이브러리:

- PyTorch
- torchvision
- timm
- OpenCV
- pandas / numpy
- scikit-learn
- matplotlib / seaborn

---

## 실행 방법

1. AI Hub에서 시설·노지 작물 질병 데이터를 다운로드합니다.
2. `preprocessing/config.py`에서 로컬 데이터 경로를 실행 환경에 맞게 수정합니다.
3. `preprocessing/`의 전처리 유틸리티를 사용하여 bbox crop, resize, split, cap을 적용합니다.
4. 최종 학습 및 평가는 아래 노트북에서 수행합니다.

```text
notebook/crop_disease_notebook_num6_last.ipynb
```

노트북에는 다음 과정이 포함되어 있습니다.

- EfficientNet-B0 baseline 학습
- EfficientNet-B3 + CBAM 최종 모델 학습
- WeightedRandomSampler 및 FocalLoss 적용
- Soft Labeling 적용
- 시설/노지 분리 평가
- held-out 5그룹 평가
- Confusion Matrix 및 Grad-CAM 분석

---

## 학습 설정

| 항목 | 값 |
|---|---:|
| Epoch | 35 |
| Early stopping patience | 7 |
| Freeze epoch | 7 |
| Learning rate | 5e-4 |
| Weight decay | 5e-4 |
| Dropout | 0.5 |
| Batch size | 64 |
| Train samples per epoch | 60,000 |
| FocalLoss gamma | 3.0 |

WeightedRandomSampler는 train DataLoader에만 적용하며, validation/test/held-out 평가에는 적용하지 않습니다.

---

## 참고사항

- 원본 데이터, 모델 가중치, 발표자료, 보고서, 중간 실험 노트북은 저장소에서 제외했습니다.
- 실행 환경에 따라 노트북 내부 경로 수정이 필요할 수 있습니다.
- 최종 성능 수치는 `crop_disease_notebook_num6_last.ipynb` 실행 결과 기준입니다.

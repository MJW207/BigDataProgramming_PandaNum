"""
model.py — EfficientNet-B0 Fine-tuning 기반 작물 질병 진행단계 분류 모델

■ 이전 버전 대비 변경사항
  - From Scratch → ImageNet 사전학습 가중치 (torchvision) Fine-tuning
  - Multi-task (Risk + Env) → 단일 태스크 (Risk 4클래스만)
  - Env 보조 헤드 완전 제거
  - CBAM (Convolutional Block Attention Module) 추가

■ 클래스: 0=정상, 1=초기, 2=중기, 3=말기

■ 2단계 학습 지원
  freeze_features()   → Phase 1: CBAM + head만 학습
  unfreeze_features() → Phase 2: 전체 fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ─────────────────────────────────────────
# CBAM
# ─────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """채널 어텐션 → 공간 어텐션 순차 적용. 병변 영역 강조."""
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(in_ch, reduction)
        self.spatial  = SpatialAttention(kernel_size=7)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────
# EfficientNet-B0 + CBAM Fine-tuning 모델
# ─────────────────────────────────────────

class EfficientNetB0CropDisease(nn.Module):
    """
    EfficientNet-B0 (ImageNet pretrained) + CBAM + Classifier Head

    구조:
      EfficientNet-B0 features → (B, 1280, 7, 7)
        → CBAM (병변 어텐션)
        → AdaptiveAvgPool2d(1) → Flatten
        → Dropout → Linear(num_classes)
    """

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3):
        super().__init__()

        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features       # 출력: (B, 1280, 7, 7)
        self.cbam     = CBAM(in_ch=1280, reduction=16)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        #self.dropout  = nn.Dropout(p=dropout_rate)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),        # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
        '''
        self.fc = nn.Linear(1280, num_classes)
        nn.init.normal_(self.fc.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc.bias)
        '''
        nn.init.normal_(self.fc[1].weight, 0.0, 0.01)   # 첫 번째 Linear (1280→512)
        nn.init.zeros_(self.fc[1].bias)
        nn.init.normal_(self.fc[5].weight, 0.0, 0.01)   # 마지막 Linear (512→num_classes)
        nn.init.zeros_(self.fc[5].bias)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # (B, 1280, 7, 7)
        x = self.cbam(x)        # 어텐션
        x = self.pool(x)        # (B, 1280, 1, 1)
        x = x.flatten(1)        # (B, 1280)
        #x = self.dropout(x)
        return self.fc(x)       # (B, num_classes)

    def freeze_features(self):
        """Phase 1: features 동결, CBAM+fc만 학습."""
        for p in self.features.parameters():
            p.requires_grad = False
        for p in list(self.cbam.parameters()) + list(self.fc.parameters()):
            p.requires_grad = True

    def unfreeze_features(self):
        """Phase 2: 전체 fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────
# 동작 검증
# ─────────────────────────────────────────

if __name__ == "__main__":
    model = EfficientNetB0CropDisease(num_classes=4, dropout_rate=0.3)

    print("=== 파라미터 ===")
    print(f"  총              : {model.total_params():,}")
    print(f"  학습 가능 (전체) : {model.trainable_params():,}")

    model.freeze_features()
    print(f"  학습 가능 (Phase1 freeze): {model.trainable_params():,}")

    model.unfreeze_features()
    print(f"  학습 가능 (Phase2 전체)  : {model.trainable_params():,}")

    print("\n=== Forward pass ===")
    out = model(torch.randn(2, 3, 224, 224))
    print(f"  출력 shape: {out.shape}")   # (2, 4)
    assert out.shape == (2, 4)
    print("  ✔ 정상 동작")

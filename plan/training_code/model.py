"""
model.py — EfficientNet-Bx + CBAM 모델

■ 설계 기반: project_design.md §4.2, §4.3, §4.4

■ 지원 버전: B0 (1280), B3 (1536), B4 (1792)

■ 구조
  EfficientNet-Bx (ImageNet 사전학습)
    → CBAM (채널 어텐션 + 공간 어텐션)
    → AdaptiveAvgPool2d(1) → Flatten
    → Dropout(0.5) → Linear(d→512) → BN → ReLU → Dropout → Linear(512→4)
    → 출력: 정상(0) / 초기(1) / 중기(2) / 말기(3)

■ 2단계 학습 지원
  freeze_features()   → Phase 1: CBAM + FC만 학습
  unfreeze_features() → Phase 2: 전체 Fine-tuning
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """채널 어텐션 → 공간 어텐션. 병변 영역 집중."""
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(in_ch, reduction)
        self.spatial  = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────
# 모델 빌더
# ─────────────────────────────────────────

_EFFICIENTNET_REGISTRY = {
    "B0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "B3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
    "B4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
}


class EfficientNetCropDisease(nn.Module):
    """
    EfficientNet-Bx + CBAM + 4-class Classifier.

    Args:
        version      : 'B0' | 'B3' | 'B4'
        num_classes  : 출력 클래스 수 (기본 4)
        dropout_rate : FC 레이어 드롭아웃 (기본 0.4)
    """

    def __init__(
        self,
        version: str = "B0",
        num_classes: int = 4,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        if version not in _EFFICIENTNET_REGISTRY:
            raise ValueError(f"지원 버전: {list(_EFFICIENTNET_REGISTRY)}")

        build_fn, weights, feat_dim = _EFFICIENTNET_REGISTRY[version]
        base = build_fn(weights=weights)

        self.version  = version
        self.feat_dim = feat_dim
        self.features = base.features          # (B, feat_dim, H, W)
        self.cbam     = CBAM(in_ch=feat_dim, reduction=16)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        # FC 가중치 초기화
        nn.init.normal_(self.fc[1].weight, 0.0, 0.01)
        nn.init.zeros_(self.fc[1].bias)
        nn.init.normal_(self.fc[5].weight, 0.0, 0.01)
        nn.init.zeros_(self.fc[5].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B, feat_dim, H, W)
        x = self.cbam(x)
        x = self.pool(x)        # (B, feat_dim, 1, 1)
        x = x.flatten(1)        # (B, feat_dim)
        return self.fc(x)       # (B, num_classes)

    # ── 피처 추출용 (Feature Extraction 단계) ──

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """CBAM 후 풀링까지 적용한 벡터 반환. BL-1 Feature Extraction용."""
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x.flatten(1)     # (B, feat_dim)

    # ── 2단계 학습 지원 ──

    def freeze_features(self):
        """Phase 1: features 동결, CBAM + FC만 학습."""
        for p in self.features.parameters():
            p.requires_grad = False
        for p in list(self.cbam.parameters()) + list(self.fc.parameters()):
            p.requires_grad = True

    def unfreeze_features(self):
        """Phase 2: 전체 Fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def phase2_param_groups(self, base_lr: float) -> list:
        """
        Phase 2 레이어별 차등 학습률 (project_design.md §4.4).

        features 하위층 × 0.1
        features 상위층 × 0.2
        CBAM         × 0.5
        FC Layer     × 1.0
        """
        n = len(self.features)
        lower_half = list(self.features[: n // 2].parameters())
        upper_half = list(self.features[n // 2 :].parameters())
        return [
            {'params': lower_half,              'lr': base_lr * 0.1},
            {'params': upper_half,              'lr': base_lr * 0.2},
            {'params': self.cbam.parameters(),  'lr': base_lr * 0.5},
            {'params': self.fc.parameters(),    'lr': base_lr * 1.0},
        ]

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────

def build_model(
    version: str = "B0",
    num_classes: int = 4,
    dropout_rate: float = 0.4,
    device: str = "cuda",
) -> EfficientNetCropDisease:
    model = EfficientNetCropDisease(version, num_classes, dropout_rate)
    return model.to(device)


if __name__ == "__main__":
    for ver in ["B0", "B3", "B4"]:
        m = EfficientNetCropDisease(version=ver)
        out = m(torch.randn(2, 3, 224, 224) if ver == "B0" else
                torch.randn(2, 3, 300, 300) if ver == "B3" else
                torch.randn(2, 3, 380, 380))
        print(f"{ver}: 출력 {out.shape}  "
              f"feat_dim={m.feat_dim}  "
              f"params={m.total_params():,}")
        assert out.shape == (2, 4)
    print("✓ 모든 버전 정상 동작")

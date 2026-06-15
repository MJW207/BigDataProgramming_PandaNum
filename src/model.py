"""
model.py v3 — EfficientNet B0/B3/B4 + CBAM

■ 변경사항
  - B3, B4 버전 지원 (EfficientNet_B3_Weights, EfficientNet_B4_Weights)
  - 버전별 feature 출력 채널 수 자동 처리 (B0=1280, B3=1536, B4=1792)
  - model_ver 인자로 버전 선택
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)


# ─────────────────────────────────────────
# 버전별 설정
# ─────────────────────────────────────────

EFFICIENTNET_CONFIGS = {
    "B0": {
        "builder":    efficientnet_b0,
        "weights":    EfficientNet_B0_Weights.IMAGENET1K_V1,
        "out_ch":     1280,
        "input_size": 224,
    },
    "B3": {
        "builder":    efficientnet_b3,
        "weights":    EfficientNet_B3_Weights.IMAGENET1K_V1,
        "out_ch":     1536,
        "input_size": 300,
    },
    "B4": {
        "builder":    efficientnet_b4,
        "weights":    EfficientNet_B4_Weights.IMAGENET1K_V1,
        "out_ch":     1792,
        "input_size": 380,
    },
}


# ─────────────────────────────────────────
# CBAM
# ─────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        mid = max(in_ch // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_ch, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(in_ch, reduction)
        self.spatial  = SpatialAttention(kernel_size=7)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────
# EfficientNet-Bx + CBAM
# ─────────────────────────────────────────

class EfficientNetCropDisease(nn.Module):
    """
    EfficientNet B0/B3/B4 + CBAM + MLP Classifier

    구조:
      EfficientNet-Bx features → (B, d, H, H)
        → CBAM
        → AdaptiveAvgPool2d(1) → Flatten → (B, d)
        → Dropout(0.5) → Linear(d→512) → BN → ReLU
        → Dropout(dropout) → Linear(512→4)

    버전별 d:
      B0: 1280 / B3: 1536 / B4: 1792
    """

    def __init__(self,
                 num_classes:  int   = 4,
                 dropout_rate: float = 0.4,
                 model_ver:    str   = "B0"):
        super().__init__()

        if model_ver not in EFFICIENTNET_CONFIGS:
            raise ValueError(f"model_ver는 B0/B3/B4 중 하나: {model_ver}")

        cfg         = EFFICIENTNET_CONFIGS[model_ver]
        self.model_ver = model_ver
        base        = cfg["builder"](weights=cfg["weights"])
        out_ch      = cfg["out_ch"]

        self.features = base.features
        self.cbam     = CBAM(in_ch=out_ch, reduction=16)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_ch, 512),    # fc[1]
            nn.BatchNorm1d(512),       # fc[2]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),  # fc[5]
        )

        # 초기화
        nn.init.normal_(self.fc[1].weight, 0.0, 0.01)
        nn.init.zeros_(self.fc[1].bias)
        nn.init.normal_(self.fc[5].weight, 0.0, 0.01)
        nn.init.zeros_(self.fc[5].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def freeze_features(self):
        """Phase 1: features 동결, CBAM + fc만 학습."""
        for p in self.features.parameters():
            p.requires_grad = False
        for p in list(self.cbam.parameters()) + list(self.fc.parameters()):
            p.requires_grad = True

    def unfreeze_features(self):
        """Phase 2: 전체 fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def get_diff_lr_params(self, base_lr: float) -> list:
        """Phase 2 차등 lr 파라미터 그룹 반환."""
        return [
            {"params": self.features[:4].parameters(), "lr": base_lr * 0.1},
            {"params": self.features[4:].parameters(), "lr": base_lr * 0.2},
            {"params": self.cbam.parameters(),         "lr": base_lr * 0.5},
            {"params": self.fc.parameters(),           "lr": base_lr},
        ]

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def out_ch(self) -> int:
        return EFFICIENTNET_CONFIGS[self.model_ver]["out_ch"]


# ─────────────────────────────────────────
# 하위 호환 별칭 (기존 코드 호환)
# ─────────────────────────────────────────

def EfficientNetB0CropDisease(num_classes=4, dropout_rate=0.4):
    return EfficientNetCropDisease(num_classes, dropout_rate, "B0")


# ─────────────────────────────────────────
# Soft Labeling Loss
# ─────────────────────────────────────────

class SoftLabelLoss(nn.Module):
    """
    진행단계 순서를 반영한 Soft Label 손실함수.
    인접 단계에 α 확률 분산, 비인접 단계는 0 유지.

    예시 (α=0.10):
      중기(2) = [0.00, 0.10, 0.80, 0.10]
      초기(1) = [0.10, 0.80, 0.10, 0.00]
      정상(0) = [0.80, 0.20, 0.00, 0.00]
      말기(3) = [0.00, 0.00, 0.20, 0.80]

    focal_gamma > 0 이면 Focal Loss와 병행 (쉬운 샘플 억제).
    """

    def __init__(self,
                 num_classes:  int   = 4,
                 alpha:        float = 0.10,
                 focal_gamma:  float = 0.0):
        super().__init__()
        self.num_classes  = num_classes
        self.alpha        = alpha
        self.focal_gamma  = focal_gamma

        # 소프트 라벨 행렬 사전 계산
        mat = torch.zeros(num_classes, num_classes)
        for c in range(num_classes):
            mat[c, c] = 1.0 - alpha
            if c > 0:
                mat[c, c - 1] = alpha / 2
            if c < num_classes - 1:
                mat[c, c + 1] = alpha / 2
            # 경계 클래스: 남은 확률을 자신에게 귀속
            mat[c] = mat[c] / mat[c].sum()
        self.register_buffer("soft_label_matrix", mat)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft_targets = self.soft_label_matrix[targets]   # (B, C)
        log_probs    = F.log_softmax(logits, dim=1)      # (B, C)

        # Cross-Entropy with soft labels
        loss = -(soft_targets * log_probs).sum(dim=1)    # (B,)

        # Focal 가중치 (선택)
        if self.focal_gamma > 0:
            probs = log_probs.exp()
            pt    = (probs * soft_targets).sum(dim=1)
            loss  = ((1 - pt) ** self.focal_gamma) * loss

        return loss.mean()


# ─────────────────────────────────────────
# 동작 검증
# ─────────────────────────────────────────

if __name__ == "__main__":
    for ver in ["B0", "B3", "B4"]:
        cfg   = EFFICIENTNET_CONFIGS[ver]
        model = EfficientNetCropDisease(model_ver=ver)
        size  = cfg["input_size"]
        out   = model(torch.randn(2, 3, size, size))
        print(f"Efficient{ver}: 입력({size}x{size}) → 출력{out.shape} "
              f"| 파라미터 {model.total_params():,}")
        assert out.shape == (2, 4)

    # Soft Label Loss 동작 확인
    loss_fn = SoftLabelLoss(alpha=0.10, focal_gamma=2.0)
    logits  = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 2, 3])
    print(f"\nSoftLabelLoss: {loss_fn(logits, targets).item():.4f}")
    print("✔ 모든 검증 통과")

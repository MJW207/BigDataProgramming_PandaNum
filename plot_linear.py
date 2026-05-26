"""
plot_linear.py — train_linear.py 학습 결과 시각화

train_linear.py에 통합하여 사용:
  from plot_linear import LinearMetricsLogger, plot_linear_results

  logger = LinearMetricsLogger()
  for epoch in range(1, epochs + 1):
      ...
      logger.update(epoch, val_f1, per_class_f1)
  plot_linear_results(logger, save_dir="./checkpoints")
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RISK_CLASSES = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}

COLORS = {
    "macro": "#2196F3",
    "정상":  "#4CAF50",
    "초기":  "#FFC107",
    "중기":  "#FF5722",
    "말기":  "#9C27B0",
    "lr":    "#607D8B",
}
NAMERE = {
    "정상":   "Norm.",   # 초록
    "초기":   "Early",   # 노랑
    "중기":   "Mid.",   # 주황빨강
    "말기":   "Term.",

}

# ─────────────────────────────────────────
# Logger
# ─────────────────────────────────────────

class LinearMetricsLogger:
    def __init__(self):
        self.epochs       = []
        self.val_f1       = []
        self.per_class_f1 = {k: [] for k in RISK_CLASSES}
        self.lr           = []
        self.train_loss   = []   # 추가
        self.val_loss     = []   # 추가

    def update(self, epoch: int, val_f1: float,
               per_class_f1, lr_now: float = 0.0,
               train_loss: float = 0.0, val_loss: float = 0.0):
        self.epochs.append(epoch)
        self.val_f1.append(val_f1)
        self.lr.append(lr_now)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        for k in RISK_CLASSES:
            self.per_class_f1[k].append(float(per_class_f1[k]))

    @property
    def best_epoch(self) -> int:
        return self.epochs[int(np.argmax(self.val_f1))]

    @property
    def best_f1(self) -> float:
        return max(self.val_f1)


# ─────────────────────────────────────────
# 플롯 함수
# ─────────────────────────────────────────

def _set_style(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if ylim:
        ax.set_ylim(*ylim)


def _draw_best_line(ax, logger: LinearMetricsLogger):
    ax.axvline(x=logger.best_epoch, color="#E0E0E0",
               linewidth=1.5, linestyle="--",
               label=f"Best epoch ({logger.best_epoch})")


def plot_loss(ax, logger: LinearMetricsLogger):
    """Train / Val Loss 곡선"""
    if not logger.train_loss or all(v == 0 for v in logger.train_loss):
        ax.text(0.5, 0.5, "No Loss Data",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax.transAxes)
        ax.set_title("Loss Curve", fontsize=13, fontweight="bold", pad=10)
        return
    ax.plot(logger.epochs, logger.train_loss,
            color="#2196F3", label="Train Loss", linewidth=2)
    ax.plot(logger.epochs, logger.val_loss,
            color="#F44336", label="Val Loss",   linewidth=2)
    _draw_best_line(ax, logger)
    _set_style(ax, "Loss Curve", "Epoch", "Loss")


def plot_macro_f1(ax, logger: LinearMetricsLogger):
    ax.plot(logger.epochs, logger.val_f1,
            color=COLORS["macro"], label="Val Macro-F1", linewidth=2)
    best_idx = int(np.argmax(logger.val_f1))
    ax.scatter(logger.epochs[best_idx], logger.val_f1[best_idx],
               color=COLORS["macro"], s=120, zorder=5,
               label=f"Best: {logger.best_f1:.4f}")
    _draw_best_line(ax, logger)
    _set_style(ax, "Val Macro-F1 Curve", "Epoch", "Macro-F1", ylim=(0, 1))


def plot_per_class_f1(ax, logger: LinearMetricsLogger):
    for k, name in RISK_CLASSES.items():
        ax.plot(logger.epochs, logger.per_class_f1[k],
                color=COLORS[name], label=NAMERE[name],
                linewidth=2, marker="o", markersize=3)
    _draw_best_line(ax, logger)
    _set_style(ax, "Val F1 by Class", "Epoch", "F1", ylim=(0, 1))


def plot_lr(ax, logger: LinearMetricsLogger):
    if all(v == 0 for v in logger.lr):
        ax.text(0.5, 0.5, "No LR Data",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax.transAxes)
        ax.set_title("Learning Rate", fontsize=13, fontweight="bold", pad=10)
        return
    ax.semilogy(logger.epochs, logger.lr,
                color=COLORS["lr"], label="Learning Rate", linewidth=2)
    _draw_best_line(ax, logger)
    _set_style(ax, "Learning Rate (log scale)", "Epoch", "LR")


def plot_best_bar(ax, logger: LinearMetricsLogger):
    """Best 에폭 클래스별 F1 막대그래프"""
    best_idx = int(np.argmax(logger.val_f1))
    names    = list(RISK_CLASSES.values())
    values   = [logger.per_class_f1[k][best_idx] for k in RISK_CLASSES]
    colors   = [COLORS[n] for n in names]
    for index, name in enumerate(names):
        names[index] = NAMERE[name]

    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    ax.axhline(y=logger.best_f1, color=COLORS["macro"],
               linewidth=1.5, linestyle="--",
               label=f"Macro-F1: {logger.best_f1:.4f}")
    ax.set_ylim(0, 1.1)
    _set_style(ax,
               f"Best Epoch ({logger.best_epoch}) F1 by Class",
               "Class", "F1")


def plot_confusion(ax, all_labels, all_preds):
    cm   = confusion_matrix(all_labels, all_preds)
    names    = list(RISK_CLASSES.values())
    for index, name in enumerate(names):
        names[index] = NAMERE[name]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=names
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Val, Best Epoch)",
                 fontsize=13, fontweight="bold", pad=10)


# ─────────────────────────────────────────
# 통합 플롯
# ─────────────────────────────────────────

def plot_linear_results(logger: LinearMetricsLogger,
                        save_dir: str = "./checkpoints",
                        all_labels: Optional[List] = None,
                        all_preds:  Optional[List] = None,
                        filename: str = "training_curves_linear.png"):
    """
    Args:
        logger     : LinearMetricsLogger 인스턴스
        save_dir   : PNG 저장 경로
        all_labels : Best 에폭의 Val 정답 라벨 (Confusion Matrix용)
        all_preds  : Best 에폭의 Val 예측 라벨
        filename   : 저장 파일명
    """
    has_cm = (all_labels is not None and all_preds is not None)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Feature Extraction Training Curves  |  "
        f"Best Val Macro-F1: {logger.best_f1:.4f} (Epoch {logger.best_epoch})",
        fontsize=15, fontweight="bold", y=1.01
    )

    plot_loss(axes[0][0], logger)          # ← Loss 곡선 (추가)
    plot_macro_f1(axes[0][1], logger)
    plot_per_class_f1(axes[0][2], logger)
    plot_best_bar(axes[1][0], logger)
    plot_lr(axes[1][1], logger)

    if has_cm:
        plot_confusion(axes[1][2], all_labels, all_preds)
    else:
        axes[1][2].axis("off")
        axes[1][2].text(0.5, 0.5, "Confusion Matrix\n(best_labels/preds 미전달)",
                        ha="center", va="center",
                        fontsize=11, color="gray")

    ax_txt = axes[1][2] if not has_cm else None
    if ax_txt is None:
        # Confusion Matrix가 있으면 요약 텍스트 표시 공간 없음 → 생략
        pass
    else:
        ax_txt.axis("off")
        best_idx   = int(np.argmax(logger.val_f1))
        tr_loss_str = (f"{logger.train_loss[best_idx]:.4f}"
                       if logger.train_loss else "N/A")
        val_loss_str= (f"{logger.val_loss[best_idx]:.4f}"
                       if logger.val_loss  else "N/A")
        summary = (
            f"■ Feature Extraction 결과 요약\n\n"
            f"  Best Epoch    : {logger.best_epoch}\n"
            f"  Macro-F1      : {logger.best_f1:.4f}\n"
            f"  Train Loss    : {tr_loss_str}\n"
            f"  Val Loss      : {val_loss_str}\n\n"
            f"  클래스별 F1 (Best Epoch)\n"
            f"  정상 : {logger.per_class_f1[0][best_idx]:.4f}\n"
            f"  초기 : {logger.per_class_f1[1][best_idx]:.4f}\n"
            f"  중기 : {logger.per_class_f1[2][best_idx]:.4f}\n"
            f"  말기 : {logger.per_class_f1[3][best_idx]:.4f}\n"
        )
        ax_txt.text(0.05, 0.95, summary,
                    transform=ax_txt.transAxes,
                    fontsize=12, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round",
                              facecolor="#F5F5F5", alpha=0.8))

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"학습 곡선 저장: {save_path}", flush=True)

"""
plot_results.py — 학습 결과 시각화

train.py 학습 루프에서 metrics를 수집하여
학습 완료 후 아래 차트를 생성한다.

  1. Train / Val Loss 곡선
  2. Train / Val Macro-F1 곡선
  3. Train - Val F1 Gap (과적합 모니터링)
  4. 클래스별 Val F1 곡선 (정상/초기/중기/말기)
  5. Learning Rate 변화 곡선
  6. 최종 Confusion Matrix

사용법:
  # train.py 내부에서 import하여 사용
  from plot_results import MetricsLogger, plot_all

  logger = MetricsLogger()
  # 에폭 루프 안에서
  logger.update(epoch, tr, val, lr_now)
  # 학습 완료 후
  plot_all(logger, save_dir="./checkpoints")
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI 없는 서버(RunPod)에서도 동작
import matplotlib.pyplot as plt
#plt.rcParams['font.family'] ='Malgun Gothic'
#plt.rcParams['axes.unicode_minus'] =False
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import RISK_CLASSES


# ─────────────────────────────────────────
# MetricsLogger — 에폭별 지표 수집
# ─────────────────────────────────────────

class MetricsLogger:
    """
    학습 루프에서 에폭마다 지표를 수집하는 컨테이너.

    사용 예:
        logger = MetricsLogger()
        for epoch in range(...):
            tr  = train_one_epoch(...)
            val = evaluate(...)
            logger.update(epoch, tr, val, lr)
        plot_all(logger, save_dir)
    """

    def __init__(self):
        self.epochs       = []
        self.tr_loss      = []
        self.val_loss     = []
        self.tr_f1        = []
        self.val_f1       = []
        self.lr           = []
        self.per_class_f1 = {k: [] for k in RISK_CLASSES}  # {0:[], 1:[], 2:[], 3:[]}
        self.phase        = []   # 1 or 2

    def update(self, epoch: int, tr: dict, val: dict,
               lr_now: float, phase: int = 2):
        """
        Args:
            epoch  : 현재 에폭 번호
            tr     : train_one_epoch 반환값 {'loss', 'f1'}
            val    : evaluate 반환값 {'loss', 'f1', 'per_class_f1'}
            lr_now : optimizer.param_groups[0]['lr']
            phase  : 1 or 2
        """
        self.epochs.append(epoch)
        self.tr_loss.append(tr["loss"])
        self.val_loss.append(val["loss"])
        self.tr_f1.append(tr["f1"])
        self.val_f1.append(val["f1"])
        self.lr.append(lr_now)
        self.phase.append(phase)

        per = val["per_class_f1"]   # numpy array [f1_0, f1_1, f1_2, f1_3]
        for k in RISK_CLASSES:
            self.per_class_f1[k].append(float(per[k]))

    @property
    def gap(self):
        return [tr - val for tr, val in zip(self.tr_f1, self.val_f1)]

    @property
    def phase2_start(self) -> Optional[int]:
        """Phase 2 시작 에폭 인덱스 반환 (없으면 None)"""
        for i, p in enumerate(self.phase):
            if p == 2:
                return i
        return None

    def save(self, path: str):
        """지표를 npy로 저장 (나중에 재플롯 가능)"""
        np.save(path, {
            "epochs":       self.epochs,
            "tr_loss":      self.tr_loss,
            "val_loss":     self.val_loss,
            "tr_f1":        self.tr_f1,
            "val_f1":       self.val_f1,
            "lr":           self.lr,
            "per_class_f1": self.per_class_f1,
            "phase":        self.phase,
        })


# ─────────────────────────────────────────
# 공통 스타일
# ─────────────────────────────────────────

COLORS = {
    "train":  "#2196F3",   # 파랑
    "val":    "#F44336",   # 빨강
    "gap":    "#FF9800",   # 주황
    "정상":   "#4CAF50",   # 초록
    "초기":   "#FFC107",   # 노랑
    "중기":   "#FF5722",   # 주황빨강
    "말기":   "#9C27B0",   # 보라
    "lr":     "#607D8B",   # 회색
    "phase":  "#E0E0E0",   # 연회색 (Phase 구분선)
}

NAMERE = {
    "정상":   "Norm.",   # 초록
    "초기":   "Early",   # 노랑
    "중기":   "Mid.",   # 주황빨강
    "말기":   "Term.",

}

def _set_style(ax, title: str, xlabel: str, ylabel: str,
               ylim: Optional[tuple] = None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if ylim:
        ax.set_ylim(*ylim)


def _draw_phase_line(ax, logger: MetricsLogger):
    """Phase 1→2 전환 에폭에 수직선 표시"""
    p2 = logger.phase2_start
    if p2 is not None and p2 > 0:
        ax.axvline(x=logger.epochs[p2], color=COLORS["phase"],
                   linewidth=1.5, linestyle="--", label="Phase 2 Start")


# ─────────────────────────────────────────
# 개별 플롯 함수
# ─────────────────────────────────────────

def plot_loss(ax, logger: MetricsLogger):
    ax.plot(logger.epochs, logger.tr_loss,
            color=COLORS["train"], label="Train Loss", linewidth=2)
    ax.plot(logger.epochs, logger.val_loss,
            color=COLORS["val"],   label="Val Loss",   linewidth=2)
    _draw_phase_line(ax, logger)
    _set_style(ax, "Loss Curve", "Epoch", "Loss")


def plot_f1(ax, logger: MetricsLogger):
    ax.plot(logger.epochs, logger.tr_f1,
            color=COLORS["train"], label="Train Macro-F1", linewidth=2)
    ax.plot(logger.epochs, logger.val_f1,
            color=COLORS["val"],   label="Val Macro-F1",   linewidth=2)
    # Best Val F1 표시
    best_idx = int(np.argmax(logger.val_f1))
    ax.scatter(logger.epochs[best_idx], logger.val_f1[best_idx],
               color=COLORS["val"], s=100, zorder=5,
               label=f"Best Val F1: {logger.val_f1[best_idx]:.4f}")
    _draw_phase_line(ax, logger)
    _set_style(ax, "Macro-F1 Curve", "Epoch", "Macro-F1", ylim=(0, 1))


def plot_gap(ax, logger: MetricsLogger):
    gap = logger.gap
    ax.plot(logger.epochs, gap,
            color=COLORS["gap"], label="Train-Val F1 Gap", linewidth=2)
    ax.axhline(y=0.15, color="red", linewidth=1,
               linestyle=":", label="Overfit Threshold (0.15)")
    ax.fill_between(logger.epochs, gap, 0.15,
                    where=[g > 0.15 for g in gap],
                    alpha=0.2, color="red", label="Overfit Range")
    _draw_phase_line(ax, logger)
    _set_style(ax, "Train-Val F1 Gap (Overfit Monitoring)",
               "Epoch", "Gap", ylim=(0, max(gap) * 1.2))


def plot_per_class_f1(ax, logger: MetricsLogger):
    for k, name in RISK_CLASSES.items():
        ax.plot(logger.epochs, logger.per_class_f1[k],
                color=COLORS[name], label=NAMERE[name], linewidth=2, marker="o",
                markersize=3)
    _draw_phase_line(ax, logger)
    _set_style(ax, "Val F1 by Class", "Epoch", "F1", ylim=(0, 1))


def plot_lr(ax, logger: MetricsLogger):
    ax.semilogy(logger.epochs, logger.lr,
                color=COLORS["lr"], label="Learning Rate", linewidth=2)
    _draw_phase_line(ax, logger)
    _set_style(ax, "Learning Rate Curve", "Epoch", "LR (log scale)")


def plot_confusion_matrix(ax, all_labels, all_preds):
    """
    최종 모델의 Confusion Matrix.
    evaluate() 반환값에서 labels/preds를 수집해서 전달.
    """
    cm      = confusion_matrix(all_labels, all_preds)
    names   = list(RISK_CLASSES.values())
    for index, name in enumerate(names):
        names[index] = NAMERE[name]
    disp    = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Val)", fontsize=13,
                 fontweight="bold", pad=10)


# ─────────────────────────────────────────
# 통합 플롯
# ─────────────────────────────────────────

def plot_all(logger: MetricsLogger,
             save_dir: str = "./checkpoints",
             all_labels=None,
             all_preds=None):
    """
    모든 차트를 하나의 Figure에 그리고 PNG로 저장.

    Args:
        logger     : MetricsLogger 인스턴스
        save_dir   : 저장 디렉터리
        all_labels : 최종 Val 정답 라벨 리스트 (Confusion Matrix용, 없으면 생략)
        all_preds  : 최종 Val 예측 라벨 리스트
    """
    has_cm  = (all_labels is not None and all_preds is not None)
    n_plots = 6 if has_cm else 5
    ncols   = 3
    nrows   = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    fig.suptitle("Result Visualization", fontsize=16, fontweight="bold", y=1.01)

    plot_loss(axes[0][0], logger)
    plot_f1(axes[0][1], logger)
    plot_gap(axes[0][2], logger)
    plot_per_class_f1(axes[1][0], logger)
    plot_lr(axes[1][1], logger)

    if has_cm:
        plot_confusion_matrix(axes[1][2], all_labels, all_preds)
    else:
        axes[1][2].axis("off")
        axes[1][2].text(0.5, 0.5,
                        "Confusion Matrix\n(no all_labels/all_preds)",
                        ha="center", va="center", fontsize=11,
                        color="gray")

    plt.tight_layout()

    save_path = Path(save_dir) / "training_curves.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"학습 곡선 저장: {save_path}", flush=True)

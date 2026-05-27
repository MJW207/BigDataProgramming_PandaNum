"""
crop_utils.py — bbox 크롭 + 30% 패딩 + Letterbox resize

원본 이미지(3024×3024 등)를 512×512 JPEG로 변환.
"""

import io
from typing import Optional, Tuple

import cv2
import numpy as np

from config import TARGET_SIZE, PADDING_RATIO


# ─────────────────────────────────────────
# bbox 크롭 + 패딩
# ─────────────────────────────────────────

def crop_with_padding(img: np.ndarray, bbox: dict, padding_ratio: float = PADDING_RATIO) -> np.ndarray:
    """
    bbox 기준 크롭 + 30% 패딩.
    bbox: {xtl, ytl, xbr, ybr} (원본 이미지 좌표계)
    img: BGR or RGB numpy (H, W, 3)
    """
    h, w = img.shape[:2]
    xtl, ytl = int(bbox["xtl"]), int(bbox["ytl"])
    xbr, ybr = int(bbox["xbr"]), int(bbox["ybr"])

    bw = max(0, xbr - xtl)
    bh = max(0, ybr - ytl)

    # 30% 패딩
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, xtl - pad_x)
    y1 = max(0, ytl - pad_y)
    x2 = min(w, xbr + pad_x)
    y2 = min(h, ybr + pad_y)

    # 크롭 결과가 너무 작으면 원본 사용
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return img

    return img[y1:y2, x1:x2]


# ─────────────────────────────────────────
# Letterbox Resize
# ─────────────────────────────────────────

def letterbox_resize(img: np.ndarray, target: int = TARGET_SIZE,
                     pad_value: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """
    종횡비 유지하면서 target×target 으로 resize. 빈 공간은 회색(114)으로 패딩.
    cv2.INTER_LANCZOS4 사용 (고품질).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target, target, 3), pad_value, dtype=np.uint8)

    scale = min(target / h, target / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 패딩
    pad_top    = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left   = (target - new_w) // 2
    pad_right  = target - new_w - pad_left

    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_value
    )


# ─────────────────────────────────────────
# 통합 처리
# ─────────────────────────────────────────

def preprocess_image(img_bytes: bytes, bbox: Optional[dict] = None) -> Optional[bytes]:
    """
    원본 이미지 bytes → 전처리 후 JPEG bytes 반환.
    실패 시 None.

    1. 이미지 디코딩
    2. bbox 있으면 크롭 + 패딩
    3. Letterbox resize 512×512
    4. JPEG 인코딩 (품질 90)
    """
    from config import JPEG_QUALITY

    # 디코딩
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # bbox 크롭
    if bbox is not None:
        img = crop_with_padding(img, bbox)

    # Letterbox resize
    img = letterbox_resize(img, target=TARGET_SIZE)

    # JPEG 인코딩
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return enc.tobytes()


if __name__ == "__main__":
    # 간단한 동작 확인
    import sys
    if len(sys.argv) < 2:
        print("사용: python crop_utils.py <원본_이미지_경로> [bbox_xtl,ytl,xbr,ybr]")
        sys.exit(1)

    img_path = sys.argv[1]
    bbox = None
    if len(sys.argv) >= 3:
        xtl, ytl, xbr, ybr = map(int, sys.argv[2].split(","))
        bbox = {"xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr}

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    result = preprocess_image(img_bytes, bbox)
    if result is None:
        print("❌ 처리 실패")
    else:
        out = "test_output.jpg"
        with open(out, "wb") as f:
            f.write(result)
        print(f"✓ 처리 완료 → {out} ({len(result):,} bytes)")

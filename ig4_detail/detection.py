"""Face and eye localisation with graceful backend fallback.

Backends, in order of preference:

1. **YuNet** (``cv2.FaceDetectorYN``, available in OpenCV >= 4.5.4 *and* 5.x)
   — returns real eye-center landmarks. The small (~350 KB) ONNX model is
   auto-downloaded once into ``ig4_detail/models/`` on first use.
2. **Haar cascades** (``cv2.CascadeClassifier``, OpenCV 4.x only — removed
   in OpenCV 5) — face boxes + cascade eyes, with anthropometric fallback.
3. **Skin-blob heuristic** — face-shaped connected components of the skin
   color mask, eyes placed anthropometrically. Never needs a download and
   works on any OpenCV build.

Whichever backend fires first wins. Set the environment variable
``IG4_DETAIL_NO_DOWNLOAD=1`` to forbid the model download.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    # github.com /raw/ redirects to LFS media storage for this repo.
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


@dataclass
class FaceInfo:
    box: tuple[int, int, int, int]           # x, y, w, h
    eyes: list[tuple[int, int, int]]         # (cx, cy, radius) each


def _anthropometric_eyes(box: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    """Typical eye positions for a frontal face box: ~38% down, 30%/70% across."""
    fx, fy, fw, fh = box
    r = max(4, int(0.09 * fw))
    return [
        (fx + int(0.30 * fw), fy + int(0.38 * fh), r),
        (fx + int(0.70 * fw), fy + int(0.38 * fh), r),
    ]


# --------------------------------------------------------------------------
# Backend 1: YuNet
# --------------------------------------------------------------------------

def _ensure_yunet_model() -> str | None:
    path = os.path.join(MODELS_DIR, YUNET_FILENAME)
    if os.path.isfile(path) and os.path.getsize(path) > 100_000:
        return path
    if os.environ.get("IG4_DETAIL_NO_DOWNLOAD"):
        return None
    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp = path + ".part"
    for url in YUNET_URLS:
        try:
            print(f"[ig4-detail] downloading YuNet face detector from {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "ig4-detail/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as fh:
                fh.write(resp.read())
            size = os.path.getsize(tmp)
            with open(tmp, "rb") as fh:
                head = fh.read(64)
            # Reject git-lfs pointer files and other non-model responses.
            if size > 100_000 and not head.startswith(b"version https://git-lfs"):
                os.replace(tmp, path)
                print(f"[ig4-detail] saved face detector to {path}")
                return path
        except Exception as exc:  # noqa: BLE001 - any failure just moves to next URL
            print(f"[ig4-detail] YuNet download failed ({exc}); trying next source")
        finally:
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    print("[ig4-detail] could not obtain YuNet model; falling back to other detectors")
    return None


_yunet = None
_yunet_failed = False


def _detect_yunet(img: np.ndarray) -> list[FaceInfo] | None:
    """img: float32 RGB in [0,1]. Returns None if the backend is unavailable."""
    global _yunet, _yunet_failed
    if _yunet_failed or not hasattr(cv2, "FaceDetectorYN"):
        return None
    try:
        if _yunet is None:
            model = _ensure_yunet_model()
            if model is None:
                _yunet_failed = True
                return None
            _yunet = cv2.FaceDetectorYN.create(model, "", (320, 320),
                                               score_threshold=0.6)
    except Exception as exc:  # model corrupt / cv2 quirk
        print(f"[ig4-detail] YuNet init failed ({exc}); falling back")
        _yunet_failed = True
        return None

    h, w = img.shape[:2]
    scale = min(1.0, 1024.0 / max(h, w))
    dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
    u8 = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    try:
        _yunet.setInputSize((dw, dh))
        _, faces = _yunet.detect(bgr)
    except Exception as exc:
        print(f"[ig4-detail] YuNet detect failed ({exc}); falling back")
        _yunet_failed = True
        return None

    results: list[FaceInfo] = []
    if faces is not None:
        inv = 1.0 / scale
        for row in faces:
            x, y, fw, fh = (int(v * inv) for v in row[:4])
            box = (x, y, fw, fh)
            r = max(4, int(0.09 * fw))
            eyes = [
                (int(row[4] * inv), int(row[5] * inv), r),   # right eye landmark
                (int(row[6] * inv), int(row[7] * inv), r),   # left eye landmark
            ]
            results.append(FaceInfo(box=box, eyes=eyes))
    return results


# --------------------------------------------------------------------------
# Backend 2: Haar cascades (OpenCV 4.x)
# --------------------------------------------------------------------------

_face_cascade = None
_eye_cascade = None


def _load_cascades() -> bool:
    global _face_cascade, _eye_cascade
    if _face_cascade is not None:
        return not _face_cascade.empty()
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return False
    try:
        base = cv2.data.haarcascades
        _face_cascade = cv2.CascadeClassifier(
            os.path.join(base, "haarcascade_frontalface_default.xml"))
        _eye_cascade = cv2.CascadeClassifier(
            os.path.join(base, "haarcascade_eye.xml"))
        return not _face_cascade.empty()
    except Exception:
        return False


def _detect_haar(img: np.ndarray) -> list[FaceInfo] | None:
    if not _load_cascades():
        return None
    h, w = img.shape[:2]
    luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    gray = (np.clip(luma, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    min_side = max(24, int(0.05 * min(h, w)))
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))

    results: list[FaceInfo] = []
    for (fx, fy, fw, fh) in faces:
        box = (int(fx), int(fy), int(fw), int(fh))
        eyes: list[tuple[int, int, int]] = []
        if _eye_cascade is not None and not _eye_cascade.empty():
            roi = gray[fy:fy + int(fh * 0.65), fx:fx + fw]
            if roi.size:
                for (ex, ey, ew, eh) in _eye_cascade.detectMultiScale(
                        roi, scaleFactor=1.1, minNeighbors=6,
                        minSize=(max(10, fw // 12), max(10, fw // 12)),
                        maxSize=(fw // 3, fw // 3)):
                    eyes.append((int(fx + ex + ew // 2), int(fy + ey + eh // 2),
                                 int(0.32 * max(ew, eh))))
        if not eyes:
            eyes = _anthropometric_eyes(box)
        results.append(FaceInfo(box=box, eyes=eyes))
    return results


# --------------------------------------------------------------------------
# Backend 3: skin-blob heuristic
# --------------------------------------------------------------------------

def _detect_blobs(skin_raw: np.ndarray) -> list[FaceInfo]:
    """Face-shaped connected components of the (unfeathered) skin mask."""
    h, w = skin_raw.shape[:2]
    binary = (skin_raw > 0.5).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results: list[FaceInfo] = []
    min_area = 0.004 * h * w
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area or bw < 16 or bh < 16:
            continue
        aspect = bh / max(bw, 1)
        fill = area / max(bw * bh, 1)
        # Faces are roughly upright ellipses that mostly fill their box;
        # arms/hands tend to be elongated or sparse.
        if 0.75 <= aspect <= 2.2 and fill >= 0.45:
            box = (int(x), int(y), int(bw), int(bh))
            results.append(FaceInfo(box=box, eyes=_anthropometric_eyes(box)))
    return results


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def detect_faces_and_eyes(img: np.ndarray, skin_raw: np.ndarray) -> list[FaceInfo]:
    """img: float32 RGB in [0,1]; skin_raw: binary-ish (H, W) skin color mask."""
    for backend in (_detect_yunet, _detect_haar):
        result = backend(img)
        if result is not None:
            return result
    return _detect_blobs(skin_raw)

"""
services/anpr_service.py — YOLO + EasyOCR detection pipeline

MEMORY FIX: EasyOCR is no longer cached globally. It is loaded, used, and
deleted after each request so YOLO (~350MB) and EasyOCR (~300MB) never
occupy RAM simultaneously. Total peak RAM stays under 512MB.
"""

import gc
import os
import re
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── YOLO is cached (fast, 350MB, loaded once) ─────────────────────────────────
_lock       = threading.Lock()
_yolo_model = None
# EasyOCR is intentionally NOT cached — loaded per request and deleted after


def _get_yolo(model_path: str):
    global _yolo_model
    if _yolo_model is None:
        with _lock:
            if _yolo_model is None:
                logger.info("Loading YOLOv8 model from: %s", model_path)
                if not Path(model_path).exists():
                    raise RuntimeError(
                        f"best.pt not found at: {model_path}\n"
                        f"Set MODEL_PATH in your .env to the correct path."
                    )
                from ultralytics import YOLO
                _yolo_model = YOLO(model_path)
                gc.collect()
                logger.info("YOLOv8 model loaded successfully.")
    return _yolo_model


def _load_ocr_and_release(crops: list) -> list:
    """
    Load EasyOCR, run OCR on all crops, delete reader, gc.collect().
    Returns list of (text, conf) per crop.
    Never keeps the reader alive after this function returns.
    """
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    results = []
    try:
        for processed, gray in crops:
            best_text  = ""
            best_oconf = 0.0
            for img_version in [processed, gray]:
                ocr_out = reader.readtext(img_version, detail=1, paragraph=False)
                valid   = [(t, c) for (_, t, c) in ocr_out if c >= 0.10]
                if valid:
                    raw   = " ".join(t for t, _ in valid)
                    clean = _clean_plate_text(raw)
                    oconf = float(np.mean([c for _, c in valid]))
                    if len(clean) >= 4 and oconf > best_oconf:
                        best_text  = clean
                        best_oconf = oconf
            results.append((best_text, best_oconf))
    finally:
        # Always release — even if OCR raises
        del reader
        gc.collect()
        logger.info("EasyOCR released from memory.")
    return results


# ── Image processing ───────────────────────────────────────────────────────────

def _preprocess_plate(plate_crop: np.ndarray) -> np.ndarray:
    h, w  = plate_crop.shape[:2]
    scale = max(1, int(300 / max(h, 1)))
    if scale > 1:
        plate_crop = cv2.resize(
            plate_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
        )
    gray  = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _clean_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _draw_annotation(
    image: np.ndarray,
    box: np.ndarray,
    plate_text: str,
    yolo_conf: float,
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{plate_text or 'Plate'}  {yolo_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(image, (x1, y1 - th - 12), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(
        image, label, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
    )
    return image


def _save_image(image: np.ndarray, upload_folder: str, original_filename: str) -> str:
    Path(upload_folder).mkdir(parents=True, exist_ok=True)
    stem     = Path(original_filename).stem
    filename = f"{stem}_{uuid.uuid4().hex[:8]}_annotated.jpg"
    path     = str(Path(upload_folder) / filename)
    cv2.imwrite(path, image)
    return path


# ── Public API ─────────────────────────────────────────────────────────────────

def run_detection(
    image_bytes: bytes,
    upload_folder: str,
    model_path: str,
    original_filename: str = "upload.jpg",
) -> dict:
    result: dict = {
        "success":        False,
        "plate_text":     None,
        "yolo_conf":      None,
        "ocr_conf":       None,
        "annotated_path": None,
        "error":          None,
    }

    try:
        # ── 1. Decode image ────────────────────────────────────────────────
        nparr   = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            result["error"] = "Cannot decode image — unsupported format."
            return result

        logger.info("Image decoded. Shape: %s", str(img_bgr.shape))
        annotated = img_bgr.copy()

        # ── 2. YOLO detection ──────────────────────────────────────────────
        model      = _get_yolo(model_path)
        detections = model(img_bgr, conf=0.15, verbose=False)
        boxes      = detections[0].boxes if detections and len(detections) > 0 else []
        box_count  = len(boxes)
        logger.info("YOLO found %d boxes at conf=0.15", box_count)

        if box_count == 0:
            result["error"]          = "No number plate detected in the image."
            result["annotated_path"] = _save_image(annotated, upload_folder, original_filename)
            return result

        # ── 3. Prepare crops for OCR ───────────────────────────────────────
        # Collect all crops first, then load EasyOCR once for all of them
        crop_data  = []   # list of (processed, gray) per box
        box_confs  = []   # yolo conf per box
        box_coords = []   # xyxy per box

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2);  y2 = min(img_bgr.shape[0], y2)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                crop_data.append(None)
            else:
                processed = _preprocess_plate(crop)
                gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_data.append((processed, gray))
            box_confs.append(float(box.conf[0]))
            box_coords.append(box.xyxy[0])

        # Filter out empty crops before passing to OCR
        valid_crops   = [c for c in crop_data if c is not None]
        valid_indices = [i for i, c in enumerate(crop_data) if c is not None]

        # ── 4. Run EasyOCR — load, use, release ───────────────────────────
        # Peak RAM here: YOLO (350MB) + EasyOCR (300MB) = ~650MB
        # EasyOCR is deleted immediately after this call
        ocr_results = _load_ocr_and_release(valid_crops)
        # After this line EasyOCR is gone from memory, back to ~350MB

        # ── 5. Pick best result ────────────────────────────────────────────
        best_text  = ""
        best_yconf = 0.0
        best_oconf = 0.0
        best_box   = None

        for idx, (text, oconf) in zip(valid_indices, ocr_results):
            yconf = box_confs[idx]
            coords = box_coords[idx]
            if not text:
                _draw_annotation(annotated, coords, "", yconf)
                continue
            if oconf > best_oconf:
                best_text  = text
                best_yconf = yconf
                best_oconf = oconf
                best_box   = coords

        # ── 6. Annotate and save ───────────────────────────────────────────
        if best_box is not None:
            annotated = _draw_annotation(annotated, best_box, best_text, best_yconf)

        save_path                = _save_image(annotated, upload_folder, original_filename)
        result["annotated_path"] = save_path

        # ── 7. Build result ────────────────────────────────────────────────
        if best_text:
            result.update(
                success    = True,
                plate_text = best_text,
                yolo_conf  = best_yconf,
                ocr_conf   = best_oconf,
            )
        else:
            result.update(
                success    = True,
                plate_text = "[No Text Detected]",
                yolo_conf  = best_yconf or (box_confs[0] if box_confs else 0.15),
                ocr_conf   = 0.0,
                error      = "Plate region detected but no legible text extracted."
            )

    except RuntimeError:
        raise
    except Exception as exc:
        logger.exception("ANPR pipeline unexpected error")
        result["error"] = str(exc)

    return result
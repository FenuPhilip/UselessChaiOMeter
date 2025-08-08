# chaiapp/views.py — final (Hough-based bubble counting, fixed 100 ml capacity)
import cv2
import numpy as np
import base64
import io
from PIL import Image
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import ChaiResult
import json
import uuid
import math
import os
from typing import Optional, Tuple

# ---------------- CONFIG ----------------
TEASPOON_ML = 5.0
GLASS_CAPACITY_ML = 100.0   # fixed glass capacity (ml) as requested
SIDE_MAX_DIM = 720          # downscale side image for speed (keeps height accuracy)
TOP_MAX_DIM = 900           # keep top reasonably large for small bubble detection
MIN_CUP_AREA_RATIO = 0.0008
MAX_CUP_ASPECT_RATIO = 4.5
DEBUG_SAVE = False
DEBUG_DIR = "/mnt/data/chai_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
# ----------------------------------------

def home(request):
    return render(request, "index.html")


def _resize_keep_aspect(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = float(max_dim) / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def decode_base64_image(data_url: str, downscale_max: Optional[int] = None) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if downscale_max:
        img = _resize_keep_aspect(img, downscale_max)
    return img


# ---- Cup detection (fast, robust) ----
def find_cup_contour(side_img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    h, w = side_img.shape[:2]
    gray = cv2.cvtColor(side_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 140)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(h * w)
    best = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_CUP_AREA_RATIO:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw <= 0 or ch <= 0:
            continue
        aspect = float(ch) / float(cw)
        if aspect > MAX_CUP_ASPECT_RATIO or ch < 8:
            continue
        score = area * min(aspect, MAX_CUP_ASPECT_RATIO)
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    pad_h = max(1, int(0.02 * ch))
    pad_w = max(1, int(0.02 * cw))
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    cw = min(side_img.shape[1] - x, cw + 2 * pad_w)
    ch = min(side_img.shape[0] - y, ch + 2 * pad_h)
    return (x, y, cw, ch)


# ---- Froth/Chai separation via LAB L-channel row analysis ----
def segment_froth_and_chai_lab(side_img: np.ndarray, cup_rect: Optional[Tuple[int,int,int,int]]):
    """
    Returns (chai_px, froth_px, cup_px) where px values are from the ROI scale used.
    We downscale the ROI vertically if it's very tall to speed up processing while keeping ratio integrity.
    """
    if cup_rect is None:
        cup_px = side_img.shape[0]
        return cup_px, 0, cup_px

    x, y, cw, ch = cup_rect
    roi = side_img[y:y+ch, x:x+cw]
    if roi.size == 0:
        cup_px = side_img.shape[0]
        return cup_px, 0, cup_px

    # downscale ROI vertical size if huge (keeps ratio)
    MAX_ROI_H = 600
    scale = 1.0
    if roi.shape[0] > MAX_ROI_H:
        scale = float(MAX_ROI_H) / float(roi.shape[0])
        roi = cv2.resize(roi, (int(roi.shape[1] * scale), MAX_ROI_H), interpolation=cv2.INTER_AREA)
    ch = roi.shape[0]

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)  # 0..255
    row_mean = L.mean(axis=1)

    # smoothing kernel proportional to height but limited
    k = max(5, min(31, (ch // 30) | 1))
    smooth = cv2.GaussianBlur(row_mean.reshape(-1, 1), (k, 1), 0).flatten()

    deriv = np.diff(smooth)  # length ch-1
    froth_bottom_row = None
    if deriv.size > 0:
        start = max(2, int(0.03 * ch))
        end = max(start+1, deriv.size - max(2, int(0.03 * ch)))
        local = deriv[start:end]
        if local.size:
            rel_min = int(np.argmin(local))
            idx = rel_min + start
            if deriv[idx] < -1.5:
                froth_bottom_row = idx + 1  # +1 due to diff shift

    # fallback heuristic if derivative fails
    if froth_bottom_row is None:
        top_blk = smooth[:max(4, int(0.06*ch))]
        bottom_blk = smooth[max(1, ch - max(6, int(0.06*ch))):]
        if top_blk.size and bottom_blk.size:
            if float(top_blk.mean()) - float(bottom_blk.mean()) > 2.5:
                pivot = float(top_blk.mean()) - (float(top_blk.mean()) - float(bottom_blk.mean())) * 0.45
                below = np.where(smooth < pivot)[0]
                if below.size:
                    froth_bottom_row = int(below[0])

    if froth_bottom_row is None or froth_bottom_row <= 0:
        froth_px = 0
        chai_px = ch
        cup_px = ch
    else:
        froth_px = int(max(0, min(ch, froth_bottom_row)))
        chai_px = int(max(0, min(ch, ch - froth_px)))
        cup_px = ch

    # if debugging, save annotated ROI
    if DEBUG_SAVE:
        vis = roi.copy()
        cv2.line(vis, (0, froth_px), (vis.shape[1]-1, froth_px), (0,0,255), 2)
        fname = os.path.join(DEBUG_DIR, f"side_debug_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(fname, vis)

    # Note: caller should use cup_px and chai_px/froth_px from this same ROI scale to compute volumes.
    return chai_px, froth_px, cup_px


# ---- Fixed-capacity volume calculation (100 ml) ----
def estimate_volumes_fixed_capacity(cup_px: int, chai_px: int, froth_px: int, capacity_ml: float = GLASS_CAPACITY_ML):
    if cup_px <= 0:
        return 0.0, 0.0, None, 0.0, 0.0

    chai_pct = round((float(chai_px) / float(cup_px)) * 100.0, 2)
    froth_pct = round((float(froth_px) / float(cup_px)) * 100.0, 2)
    px_to_ml = float(capacity_ml) / float(cup_px)
    chai_ml = float(chai_px) * px_to_ml
    froth_ml = float(froth_px) * px_to_ml

    ratio_val = round(float(chai_ml) / float(froth_ml), 2) if froth_ml > 1e-6 else None
    chai_teaspoons = round(float(chai_ml) / TEASPOON_ML, 2)

    # round ml to 2 decimals
    chai_ml = round(float(chai_ml), 2)
    froth_ml = round(float(froth_ml), 2)

    return chai_pct, froth_pct, ratio_val, chai_ml, chai_teaspoons


# ---- Rim mask helper (fast) ----
def detect_rim_mask(top_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Try to detect rim with HoughCircles tuned for rim size. If found, make an interior mask.
    Else fallback to finding the largest near-circular contour.
    """
    gray = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blur = cv2.medianBlur(gray, 7)

    # Hough for rim (rim is large)
    try:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0,
                                   minDist=min(w,h)/8,
                                   param1=100, param2=30,
                                   minRadius=int(min(w,h)*0.18),
                                   maxRadius=int(min(w,h)*0.49))
    except Exception:
        circles = None

    if circles is not None and len(circles[0]) > 0:
        c = circles[0][0]
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.95), 255, -1)
        if DEBUG_SAVE:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (cx, cy), r, (0,255,0), 2)
            fname = os.path.join(DEBUG_DIR, f"top_rim_{uuid.uuid4().hex[:8]}.jpg")
            cv2.imwrite(fname, vis)
        return mask

    # Fallback contour method
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, -1)
    # don't erode too much; keep near-rim bubbles
    if DEBUG_SAVE:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [best], -1, (0,255,0), 2)
        fname = os.path.join(DEBUG_DIR, f"top_rim_cont_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(fname, vis)
    return mask


# ---- Hough-based bubble counting (primary) ----
def count_bubbles_hough(top_img: np.ndarray) -> int:
    """
    Primary bubble detector using HoughCircles on a well-preprocessed image, restricted by rim mask.
    Dedupe overlapping circles and filter by reasonable radius range.
    """
    img = _resize_keep_aspect(top_img, TOP_MAX_DIM)
    h, w = img.shape[:2]

    # Mask to interior of rim
    mask = detect_rim_mask(img)
    if mask is None:
        # fallback to center circle mask
        cx, cy = w//2, h//2
        r = int(min(w, h) * 0.45)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

    # Preprocess for Hough: equalize and median blur (Hough works better without extreme binary morph)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.medianBlur(eq, 5)

    # Hough parameters tuned for small bubbles:
    min_dist = max(6, int(min(w, h) * 0.03))   # minimum distance between centers
    min_radius = 2
    max_radius = max(6, int(min(w, h) * 0.12))  # allow up to ~12% of min-dim

    try:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=min_dist,
                                   param1=50, param2=16,  # param2 lower -> more sensitive
                                   minRadius=min_radius, maxRadius=max_radius)
    except Exception:
        circles = None

    kept = []
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = int(round(c[0])), int(round(c[1])), int(round(c[2]))
            # center must be inside mask
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            if mask[cy, cx] == 0:
                continue
            # radius plausibility check
            if r < min_radius or r > max_radius:
                continue
            # dedupe by proximity (keep larger circle if overlap)
            dup = False
            for i, (pcx, pcy, pr) in enumerate(kept):
                if math.hypot(cx - pcx, cy - pcy) < max(3, 0.6 * (r + pr)):
                    # overlap - keep the one with larger radius confidence
                    if r > pr:
                        kept[i] = (cx, cy, r)
                    dup = True
                    break
            if not dup:
                kept.append((cx, cy, r))

    # If Hough found very few or none, try fast contour fallback (still masked)
    if len(kept) < 3:
        # adaptive fallback: threshold masked image and look for circular-ish contours
        masked = cv2.bitwise_and(eq, eq, mask=mask)
        blur2 = cv2.GaussianBlur(masked, (5,5), 0)
        _, thr = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw == 0 or ch == 0:
                continue
            r_est = max(cw, ch)/2.0
            if r_est < min_radius or r_est > max_radius:
                continue
            per = cv2.arcLength(cnt, True)
            if per <= 0:
                continue
            circ = 4.0 * math.pi * (area / (per * per))
            if circ < 0.18:  # allow some irregularity
                continue
            M = cv2.moments(cnt)
            if M.get("m00",0) == 0:
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            if mask[cy, cx] == 0:
                continue
            # dedupe vs kept
            dup = False
            for i, (pcx, pcy, pr) in enumerate(kept):
                if math.hypot(cx - pcx, cy - pcy) < max(3, 0.6 * (r_est + pr)):
                    dup = True
                    break
            if not dup:
                kept.append((cx, cy, r_est))

    # final count
    bubble_count = len(kept)

    # debug visualization
    if DEBUG_SAVE:
        vis = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        for (cx, cy, r) in kept:
            cv2.circle(vis, (int(cx), int(cy)), int(max(2, round(r))), (0,255,0), 2)
        # draw mask boundary overlay
        contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_mask, -1, (255,0,0), 2)
        fname = os.path.join(DEBUG_DIR, f"top_bubbles_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(fname, vis)

    return int(bubble_count)


# ---- Roast generation ----
def generate_roast(chai_pct: float, froth_pct: float, ratio_val: Optional[float]) -> str:
    if (ratio_val is None) and (chai_pct == 0.0):
        return "Where did the chai go? Empty glass?"
    if ratio_val is None:
        return "Very foamy (or no froth) — odd."
    if ratio_val > 30.0:
        return "All chai, no froth — where's the foam?"
    if ratio_val < 1.0:
        return "That's a foam party. Is the chai shy?"
    if 0.95 <= ratio_val <= 1.5:
        return "You're the chosen one with that perfect chai!"
    return "Decent brew — not a crime, not a miracle."


# ---- API endpoint ----
@csrf_exempt
def upload_images(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    try:
        payload = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "side_view" not in payload or "top_view" not in payload:
        return JsonResponse({"error": "side_view and top_view required"}, status=400)

    try:
        side_img = decode_base64_image(payload["side_view"], downscale_max=SIDE_MAX_DIM)
        top_img = decode_base64_image(payload["top_view"], downscale_max=TOP_MAX_DIM)
    except Exception as e:
        return JsonResponse({"error": f"Failed to decode images: {e}"}, status=400)

    # detect cup on side view
    cup_rect = find_cup_contour(side_img)
    if cup_rect is None:
        # fallback: use full image as cup rect
        h, w = side_img.shape[:2]
        cup_rect = (0, 0, w, h)

    # segment
    chai_px, froth_px, cup_px = segment_froth_and_chai_lab(side_img, cup_rect)

    # estimate volumes using fixed 100 ml capacity
    chai_pct, froth_pct, ratio_val, chai_ml, chai_tsp = estimate_volumes_fixed_capacity(
        cup_px, chai_px, froth_px, capacity_ml=GLASS_CAPACITY_ML
    )

    # bubble count using Hough
    bubble_count = count_bubbles_hough(top_img)

    # roast
    roast = generate_roast(chai_pct, froth_pct, ratio_val)

    # save result (don't fail API on DB error)
    try:
        ChaiResult.objects.create(
            device_uuid=request.COOKIES.get("device_uuid") or str(uuid.uuid4()),
            chai_height=chai_pct,
            froth_height=froth_pct,
            chai_to_froth_ratio=(ratio_val if ratio_val is not None else 0.0),
            bubble_count=bubble_count,
            roast=roast
        )
    except Exception:
        pass

    resp = {
        "chai_height": chai_pct,
        "froth_height": froth_pct,
        "ratio": ratio_val,  # numeric ratio or null
        "chai_ml": chai_ml,
        "chai_teaspoons": chai_tsp,
        "bubble_count": bubble_count,
        "roast": roast
    }
    return JsonResponse(resp)

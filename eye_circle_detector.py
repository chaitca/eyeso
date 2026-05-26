import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =========================
# 在这里修改输入和输出路径
# =========================
# 可以是图片路径，也可以是视频路径。
INPUT_PATH = r"D:\Eyeso\all\photo\003-jyy.png"

# 检测结果保存位置。
OUTPUT_DIR = r"D:\Eyeso\all\photo\result"

# 可选 ROI，格式为 "x,y,w,h"。不需要就写 None。
# 如果视频/截图里眼睛只占画面一部分，建议填写 ROI 来提高检测准确度。
ROI = None

# 视频批量检测时，每隔多少帧检测一次。
EVERY_N_FRAMES = 30

# 如果只想检测视频中的某一帧，填整数；批量检测则写 None。
FRAME_INDEX = None

# 如果知道像素到毫米的比例，填数值；不知道就写 None。
MM_PER_PIXEL = None

# 瞳孔圆半径修正系数。绿色圆偏大时可调小，例如 0.90。
PUPIL_SCALE = 0.95

# Yellow circle: iris/WHW outer boundary settings.
# If yellow circle is still too large, reduce OUTER_MAX_SCALE, for example 2.10.
# If yellow circle is too small, increase OUTER_MAX_SCALE or OUTER_MIN_SCALE slightly.
# 黄色圈偏大：把 OUTER_MAX_SCALE 调小，比如 2.10。
# 黄色圈偏小：把 OUTER_MAX_SCALE 调大，比如 2.50。
OUTER_MIN_SCALE = 1.35
OUTER_MAX_SCALE = 3.5
OUTER_CENTER_TOLERANCE = 0.35


@dataclass
class Circle:
    x: float
    y: float
    r: float
    score: float = 0.0

    @property
    def diameter(self) -> float:
        return 2.0 * self.r


def parse_roi(roi_text):
    if not roi_text:
        return None
    parts = [int(v.strip()) for v in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return tuple(parts)


def crop_roi(image, roi):
    if roi is None:
        return image, (0, 0)
    x, y, w, h = roi
    return image[y : y + h, x : x + w], (x, y)


def preprocess_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_pupil(gray):
    h, w = gray.shape
    min_area = max(80, int(h * w * 0.002))
    max_area = int(h * w * 0.30)

    # The pupil is normally the largest dark, compact region in an eye ROI.
    thresh_value = np.percentile(gray, 12)
    dark = (gray <= thresh_value).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_center = np.array([w / 2.0, h / 2.0])

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (x, y), r = cv2.minEnclosingCircle(contour)
        if r < 5:
            continue
        fill_ratio = area / (np.pi * r * r)
        center_penalty = np.linalg.norm(np.array([x, y]) - image_center) / max(h, w)
        score = circularity * 0.55 + fill_ratio * 0.55 - center_penalty * 0.35
        candidates.append(Circle(x, y, r, score))

    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def circle_gradient_score(gray, cx, cy, radius, angles):
    h, w = gray.shape
    samples_inside = []
    samples_outside = []
    for angle in angles:
        ca, sa = np.cos(angle), np.sin(angle)
        x1 = int(round(cx + (radius - 2) * ca))
        y1 = int(round(cy + (radius - 2) * sa))
        x2 = int(round(cx + (radius + 2) * ca))
        y2 = int(round(cy + (radius + 2) * sa))
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            samples_inside.append(gray[y1, x1])
            samples_outside.append(gray[y2, x2])
    if len(samples_inside) < max(12, len(angles) // 5):
        return -1.0
    return abs(float(np.mean(samples_outside)) - float(np.mean(samples_inside)))


def radial_mean(gray, cx, cy, radius, angles):
    h, w = gray.shape
    values = []
    for angle in angles:
        x = int(round(cx + radius * np.cos(angle)))
        y = int(round(cy + radius * np.sin(angle)))
        if 0 <= x < w and 0 <= y < h:
            values.append(gray[y, x])
    if len(values) < max(12, len(angles) // 5):
        return None
    return float(np.mean(values))


def refine_pupil_radius(gray, pupil):
    # The dark-region contour often lands on the pupil core. Move the radius to
    # the outer side of the dark-to-iris transition seen in the radial profile.
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    min_r = max(5, int(pupil.r * 0.65))
    max_r = int(pupil.r * 1.45)
    radii = np.arange(min_r, max_r + 1)
    means = []
    valid_radii = []

    for r in radii:
        mean = radial_mean(gray, pupil.x, pupil.y, r, angles)
        if mean is not None:
            valid_radii.append(r)
            means.append(mean)

    if len(means) < 12:
        return pupil

    means = np.array(means, dtype=np.float32)
    valid_radii = np.array(valid_radii, dtype=np.float32)
    kernel_size = min(7, len(means) if len(means) % 2 == 1 else len(means) - 1)
    if kernel_size >= 3:
        smooth = np.convolve(means, np.ones(kernel_size) / kernel_size, mode="same")
    else:
        smooth = means

    gradient = np.gradient(smooth)
    edge_start = max(3, int(len(valid_radii) * 0.15))
    edge_end = max(edge_start + 1, int(len(valid_radii) * 0.92))
    edge_idx = edge_start + int(np.argmax(gradient[edge_start:edge_end]))

    inside = np.percentile(smooth[max(0, edge_idx - 12) : edge_idx], 30)
    outside = np.percentile(smooth[edge_idx : min(len(smooth), edge_idx + 18)], 85)
    if outside <= inside:
        return pupil

    target = inside + 0.90 * (outside - inside)
    refined_r = pupil.r
    for idx in range(edge_idx, len(valid_radii)):
        if smooth[idx] >= target:
            refined_r = float(valid_radii[idx])
            break

    if pupil.r * 0.80 <= refined_r <= pupil.r * 1.35:
        return Circle(pupil.x, pupil.y, refined_r, pupil.score)
    return pupil


def iris_angles():
    # Use mostly horizontal rays. Vertical rays are often corrupted by eyelids,
    # eyelashes, and reflections, which can pull the outer circle far too large.
    right = np.deg2rad(np.linspace(-45, 45, 91))
    left = np.deg2rad(np.linspace(135, 225, 91))
    return np.concatenate([right, left])


def smooth_1d(values, window):
    if len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


def find_limbus_side_distance(gray, pupil, side):
    h, w = gray.shape
    cx, cy = pupil.x, pupil.y
    sign = 1 if side == "right" else -1
    band_half_height = max(8, int(pupil.r * 0.22))
    y1 = max(0, int(round(cy - band_half_height)))
    y2 = min(h, int(round(cy + band_half_height + 1)))

    min_d = int(max(pupil.r * 1.70, pupil.r + 24))
    max_bound = (w - cx - 3) if sign > 0 else (cx - 3)
    max_d = int(min(pupil.r * 2.70, max_bound))
    if max_d <= min_d + 20:
        return None

    distances = np.arange(min_d, max_d + 1)
    profile = []
    for distance in distances:
        x = int(round(cx + sign * distance))
        x1 = max(0, x - 2)
        x2 = min(w, x + 3)
        profile.append(float(np.median(gray[y1:y2, x1:x2])))

    profile = smooth_1d(np.array(profile, dtype=np.float32), 17)
    gap = max(6, int(pupil.r * 0.08))
    candidates = []
    for idx in range(gap, len(distances) - gap):
        inner = float(np.mean(profile[idx - gap : idx]))
        outer = float(np.mean(profile[idx : idx + gap]))
        score = outer - inner
        if score > 0:
            candidates.append((score, int(distances[idx])))

    if not candidates:
        return None

    best_score = max(score for score, _ in candidates)
    min_score = max(2.0, best_score * 0.35)
    strong = [(score, distance) for score, distance in candidates if score >= min_score]
    if not strong:
        return None

    target_distance = pupil.r * 2.25

    def candidate_score(item):
        score, distance = item
        distance_penalty = abs(distance - target_distance) * 0.04
        return score - distance_penalty

    return max(strong, key=candidate_score)


def detect_limbus_circle_from_sides(gray, pupil):
    left = find_limbus_side_distance(gray, pupil, "left")
    right = find_limbus_side_distance(gray, pupil, "right")
    if left is None and right is None:
        return None

    distances = []
    scores = []
    if left is not None:
        scores.append(left[0])
        distances.append(left[1])
    if right is not None:
        scores.append(right[0])
        distances.append(right[1])

    radius = float(np.median(distances))
    center_x = float(pupil.x)
    if left is not None and right is not None:
        left_x = pupil.x - left[1]
        right_x = pupil.x + right[1]
        center_x = float((left_x + right_x) / 2.0)
        radius = float((right_x - left_x) / 2.0)

    min_radius = pupil.r * 1.65
    max_radius = pupil.r * OUTER_MAX_SCALE
    if not (min_radius <= radius <= max_radius):
        return None

    return Circle(center_x, float(pupil.y), radius, float(np.mean(scores)))


def detect_outer_circle(
    gray,
    pupil,
    min_scale=OUTER_MIN_SCALE,
    max_scale=OUTER_MAX_SCALE,
    center_tolerance=OUTER_CENTER_TOLERANCE,
):
    h, w = gray.shape
    cx, cy = pupil.x, pupil.y
    min_r = int(max(pupil.r * min_scale, pupil.r + 12))
    max_r = int(min(max(h, w) * 0.30, pupil.r * max_scale))
    if max_r <= min_r:
        return None

    side_circle = detect_limbus_circle_from_sides(gray, pupil)
    if side_circle is not None:
        return side_circle

    edges = cv2.Canny(gray, 40, 120)
    mask = np.zeros_like(edges)
    cv2.circle(mask, (int(cx), int(cy)), max_r + 12, 255, -1)
    cv2.circle(mask, (int(cx), int(cy)), min_r - 6, 0, -1)
    edges = cv2.bitwise_and(edges, mask)

    hough = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(15, int(pupil.r)),
        param1=80,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )

    candidates = []
    if hough is not None:
        for x, y, r in np.round(hough[0]).astype(int):
            center_distance = np.hypot(x - cx, y - cy)
            if center_distance > pupil.r * center_tolerance:
                continue
            candidates.append(Circle(float(x), float(y), float(r), 0.0))

    # Fallback and refinement: score radii near the limbus/iris border.
    angles = iris_angles()
    radial_candidates = []
    for r in range(min_r, max_r + 1):
        score = circle_gradient_score(gray, cx, cy, r, angles)
        radial_candidates.append(Circle(cx, cy, float(r), score))
    if radial_candidates:
        candidates.append(max(radial_candidates, key=lambda c: c.score))

    if not candidates:
        return None

    for c in candidates:
        gradient = circle_gradient_score(gray, c.x, c.y, c.r, angles)
        center_penalty = np.hypot(c.x - cx, c.y - cy) / max(pupil.r, 1.0)
        scale = c.r / max(pupil.r, 1.0)
        scale_penalty = abs(scale - 1.85) * 2.0
        c.score = gradient - 8.0 * center_penalty - scale_penalty

    return max(candidates, key=lambda c: c.score)


def detect_eye_circles(image, roi=None, pupil_scale=0.95):
    cropped, offset = crop_roi(image, roi)
    gray = preprocess_gray(cropped)
    pupil = detect_pupil(gray)
    if pupil is None:
        return None, None
    pupil = refine_pupil_radius(gray, pupil)
    outer = detect_outer_circle(gray, pupil)
    pupil = Circle(pupil.x, pupil.y, pupil.r * pupil_scale, pupil.score)

    ox, oy = offset
    pupil = Circle(pupil.x + ox, pupil.y + oy, pupil.r, pupil.score)
    if outer is not None:
        outer = Circle(outer.x + ox, outer.y + oy, outer.r, outer.score)
    return pupil, outer


def annotate(image, pupil, outer):
    out = image.copy()
    if outer is not None:
        cv2.circle(out, (int(round(outer.x)), int(round(outer.y))), int(round(outer.r)), (0, 180, 255), 2)
        cv2.putText(
            out,
            f"outer D={outer.diameter:.1f}px",
            (int(outer.x - outer.r), max(25, int(outer.y - outer.r - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    if pupil is not None:
        cv2.circle(out, (int(round(pupil.x)), int(round(pupil.y))), int(round(pupil.r)), (0, 255, 0), 2)
        cv2.circle(out, (int(round(pupil.x)), int(round(pupil.y))), 2, (0, 255, 0), -1)
        cv2.putText(
            out,
            f"pupil D={pupil.diameter:.1f}px",
            (int(pupil.x - pupil.r), int(pupil.y + pupil.r + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def iter_inputs(input_path, every_n_frames=30, frame_index=None):
    path = Path(input_path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        yield path.stem, image
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    if frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if ok:
            yield f"{path.stem}_frame_{frame_index:06d}", frame
        cap.release()
        return

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n_frames == 0:
            yield f"{path.stem}_frame_{idx:06d}", frame
        idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Detect pupil and outer iris/ring circle diameters.")
    parser.add_argument("input", nargs="?", default=INPUT_PATH, help="Image or video path")
    parser.add_argument("--roi", default=ROI, help="Optional ROI as x,y,w,h. Strongly recommended for screenshots/videos.")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="Directory for annotated images and CSV")
    parser.add_argument("--every-n-frames", type=int, default=EVERY_N_FRAMES, help="For video batch mode")
    parser.add_argument("--frame-index", type=int, default=FRAME_INDEX, help="Detect one video frame")
    parser.add_argument("--mm-per-pixel", type=float, default=MM_PER_PIXEL, help="Optional scale for millimeter output")
    parser.add_argument(
        "--pupil-scale",
        type=float,
        default=PUPIL_SCALE,
        help="Scale detected pupil radius. Use smaller values, e.g. 0.90, if the green circle is too large.",
    )
    args = parser.parse_args()

    roi = parse_roi(args.roi)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, image in iter_inputs(args.input, args.every_n_frames, args.frame_index):
        pupil, outer = detect_eye_circles(image, roi=roi, pupil_scale=args.pupil_scale)
        annotated = annotate(image, pupil, outer)
        cv2.imwrite(str(out_dir / f"{name}_detected.png"), annotated)

        row = {
            "name": name,
            "pupil_x": "" if pupil is None else f"{pupil.x:.2f}",
            "pupil_y": "" if pupil is None else f"{pupil.y:.2f}",
            "pupil_diameter_px": "" if pupil is None else f"{pupil.diameter:.2f}",
            "outer_x": "" if outer is None else f"{outer.x:.2f}",
            "outer_y": "" if outer is None else f"{outer.y:.2f}",
            "outer_diameter_px": "" if outer is None else f"{outer.diameter:.2f}",
        }
        if args.mm_per_pixel:
            row["pupil_diameter_mm"] = "" if pupil is None else f"{pupil.diameter * args.mm_per_pixel:.3f}"
            row["outer_diameter_mm"] = "" if outer is None else f"{outer.diameter * args.mm_per_pixel:.3f}"
        rows.append(row)
        print(row)

    if rows:
        csv_path = out_dir / "diameters.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

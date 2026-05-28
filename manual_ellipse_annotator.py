import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import matplotlib

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HANDLE_RADIUS_PX = 12.0


@dataclass
class EllipseModel:
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    @property
    def major_radius(self):
        return self.width / 2.0

    @property
    def minor_radius(self):
        return self.height / 2.0

    def unit_major(self):
        theta = np.deg2rad(self.angle)
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def unit_minor(self):
        theta = np.deg2rad(self.angle + 90.0)
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def handle_positions(self):
        center = np.array([self.center_x, self.center_y], dtype=float)
        major = self.unit_major() * self.major_radius
        minor = self.unit_minor() * self.minor_radius
        return {
            "center": center,
            "major_pos": center + major,
            "major_neg": center - major,
            "minor_pos": center + minor,
            "minor_neg": center - minor,
        }

    def move(self, dx, dy):
        self.center_x += dx
        self.center_y += dy

    def drag_major_handle(self, x, y):
        center = np.array([self.center_x, self.center_y], dtype=float)
        vector = np.array([x, y], dtype=float) - center
        radius = float(np.linalg.norm(vector))
        if radius < 2.0:
            return
        self.width = max(4.0, radius * 2.0)
        self.angle = float(np.rad2deg(np.arctan2(vector[1], vector[0])))

    def drag_minor_handle(self, x, y):
        center = np.array([self.center_x, self.center_y], dtype=float)
        vector = np.array([x, y], dtype=float) - center
        radius = float(np.linalg.norm(vector))
        if radius < 2.0:
            return
        self.height = max(4.0, radius * 2.0)
        self.angle = float(np.rad2deg(np.arctan2(vector[1], vector[0])) - 90.0)


def normalize_fit(raw_ellipse):
    (cx, cy), (width, height), angle = raw_ellipse
    width = float(width)
    height = float(height)
    angle = float(angle)
    if height > width:
        width, height = height, width
        angle += 90.0
    angle = ((angle + 180.0) % 360.0) - 180.0
    return EllipseModel(float(cx), float(cy), width, height, angle)


def sort_points_around_center(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def fit_ellipse_from_points(points):
    if len(points) < 5:
        return None
    sorted_points = sort_points_around_center(points[:5])
    pts = sorted_points.reshape(-1, 1, 2)
    try:
        ellipse = normalize_fit(cv2.fitEllipseDirect(pts))
    except cv2.error:
        try:
            ellipse = normalize_fit(cv2.fitEllipse(pts))
        except cv2.error:
            return None
    if ellipse.width < 4.0 or ellipse.height < 4.0:
        return None
    point_span = sorted_points.max(axis=0) - sorted_points.min(axis=0)
    if ellipse.width < point_span[0] * 0.75 or ellipse.height < point_span[1] * 0.75:
        return None
    return ellipse


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix, image_bgr)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def choose_image_files():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filenames = filedialog.askopenfilenames(
        title="Open eye images",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return list(filenames)


class ManualEllipseAnnotator:
    def __init__(self, image_paths=None, out_dir=None):
        self.image_path = None
        self.image_paths = []
        self.image_index = -1
        self.image = None
        self.out_dir = Path(out_dir) if out_dir else None

        self.labels = ["iris", "pupil"]
        self.colors = {"iris": "#ffcc00", "pupil": "#00e676"}
        self.active_label = "iris"
        self.points = {"iris": [], "pupil": []}
        self.ellipses = {"iris": None, "pupil": None}
        self.patches = {}
        self.point_artists = {}
        self.handle_artists = {}
        self.drag_state = None
        self.closing = False

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Manual iris/pupil ellipse annotator")
        plt.subplots_adjust(bottom=0.18)
        self.status = self.fig.text(0.015, 0.035, "", fontsize=10)
        self.help_text = self.fig.text(
            0.015,
            0.01,
            "Keys: I iris | P pupil | left click add 5 points | drag handles after fit | Z undo | R reset | O open | S save | N next | B previous",
            fontsize=9,
        )

        self._make_buttons()
        self._connect_events()

        if image_paths:
            self.set_image_list(image_paths)
        else:
            self.show_empty_screen()

    def _make_buttons(self):
        button_specs = [
            ("Open Images", 0.03, 0.105, self.open_image_dialog),
            ("Prev", 0.15, 0.07, self.previous_image),
            ("Next", 0.23, 0.07, self.next_image),
            ("Iris", 0.31, 0.07, lambda _event: self.set_active("iris")),
            ("Pupil", 0.39, 0.07, lambda _event: self.set_active("pupil")),
            ("Undo", 0.47, 0.07, self.undo_point),
            ("Reset", 0.55, 0.07, self.reset_active),
            ("Save", 0.63, 0.07, self.save_outputs),
            ("Save+Next", 0.71, 0.095, self.save_and_next),
            ("Quit", 0.82, 0.07, self.quit_app),
        ]
        self.buttons = []
        for label, left, width, callback in button_specs:
            button_ax = self.fig.add_axes([left, 0.075, width, 0.055])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def show_empty_screen(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(
            0.5,
            0.56,
            "Open Image",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        self.ax.text(
            0.5,
            0.48,
            "Click Open Images below to choose one or more original eye images.",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        self.status.set_text("No image loaded. Click Open Images to choose files.")
        self.fig.canvas.draw_idle()

    def _connect_events(self):
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("key_press_event", self.on_key)
        canvas.mpl_connect("close_event", self.quit_app)

    def open_image_dialog(self, _event=None):
        filenames = choose_image_files()
        if filenames:
            self.set_image_list(filenames)

    def set_image_list(self, image_paths, start_index=0):
        self.image_paths = [
            Path(path)
            for path in image_paths
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not self.image_paths:
            self.status.set_text("No supported image files were selected.")
            self.fig.canvas.draw_idle()
            return
        self.image_index = min(max(start_index, 0), len(self.image_paths) - 1)
        self.load_image(self.image_paths[self.image_index])

    def load_image(self, image_path):
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        self.image_path = path
        self.image = read_image(path)
        self.points = {"iris": [], "pupil": []}
        self.ellipses = {"iris": None, "pupil": None}
        self.active_label = "iris"
        self.ax.clear()
        self.ax.imshow(self.image)
        title_prefix = ""
        if self.image_paths:
            title_prefix = f"[{self.image_index + 1}/{len(self.image_paths)}] "
        self.ax.set_title(f"{title_prefix}{path}")
        self.ax.set_axis_off()
        self._redraw()

    def set_active(self, label):
        if label in self.labels:
            self.active_label = label
            self._redraw()

    def on_key(self, event):
        if event.key in {"i", "I"}:
            self.set_active("iris")
        elif event.key in {"p", "P"}:
            self.set_active("pupil")
        elif event.key in {"z", "Z", "backspace"}:
            self.undo_point()
        elif event.key in {"r", "R"}:
            self.reset_active()
        elif event.key in {"o", "O"}:
            self.open_image_dialog()
        elif event.key in {"s", "S"}:
            self.save_outputs()
        elif event.key in {"n", "N"}:
            self.next_image()
        elif event.key in {"b", "B"}:
            self.previous_image()
        elif event.key in {"q", "Q", "escape"}:
            self.quit_app()

    def on_press(self, event):
        if self.image is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        hit = self._hit_test_handles(event.xdata, event.ydata)
        if hit is not None:
            label, handle_name = hit
            self.active_label = label
            self.drag_state = {
                "label": label,
                "handle": handle_name,
                "last_x": event.xdata,
                "last_y": event.ydata,
            }
            return

        if event.button == 1:
            points = self.points[self.active_label]
            if len(points) < 5:
                points.append((float(event.xdata), float(event.ydata)))
                if len(points) == 5:
                    self.ellipses[self.active_label] = fit_ellipse_from_points(points)
                self._redraw()

    def on_motion(self, event):
        if self.drag_state is None or self.image is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        label = self.drag_state["label"]
        handle_name = self.drag_state["handle"]
        ellipse = self.ellipses[label]
        if ellipse is None:
            return

        if handle_name == "center":
            dx = event.xdata - self.drag_state["last_x"]
            dy = event.ydata - self.drag_state["last_y"]
            ellipse.move(dx, dy)
            self.drag_state["last_x"] = event.xdata
            self.drag_state["last_y"] = event.ydata
        elif handle_name.startswith("major"):
            ellipse.drag_major_handle(event.xdata, event.ydata)
        elif handle_name.startswith("minor"):
            ellipse.drag_minor_handle(event.xdata, event.ydata)
        self._redraw()

    def on_release(self, _event):
        self.drag_state = None

    def undo_point(self, _event=None):
        points = self.points[self.active_label]
        if points:
            points.pop()
            self.ellipses[self.active_label] = None
        self._redraw()

    def reset_active(self, _event=None):
        self.points[self.active_label] = []
        self.ellipses[self.active_label] = None
        self._redraw()

    def next_image(self, _event=None):
        if not self.image_paths:
            return
        if self.image_index >= len(self.image_paths) - 1:
            self.status.set_text("Already at the last image.")
            self.fig.canvas.draw_idle()
            return
        self.image_index += 1
        self.load_image(self.image_paths[self.image_index])

    def previous_image(self, _event=None):
        if not self.image_paths:
            return
        if self.image_index <= 0:
            self.status.set_text("Already at the first image.")
            self.fig.canvas.draw_idle()
            return
        self.image_index -= 1
        self.load_image(self.image_paths[self.image_index])

    def _hit_test_handles(self, x, y):
        click = np.array([x, y], dtype=float)
        best = None
        best_distance = float("inf")
        for label, ellipse in self.ellipses.items():
            if ellipse is None:
                continue
            for handle_name, pos in ellipse.handle_positions().items():
                distance = float(np.linalg.norm(click - pos))
                if distance < best_distance and distance <= HANDLE_RADIUS_PX:
                    best = (label, handle_name)
                    best_distance = distance
        return best

    def _clear_artists(self):
        for collection in [self.patches, self.point_artists, self.handle_artists]:
            for artist in collection.values():
                if isinstance(artist, list):
                    for item in artist:
                        try:
                            item.remove()
                        except ValueError:
                            pass
                else:
                    try:
                        artist.remove()
                    except ValueError:
                        pass
            collection.clear()

    def _redraw(self):
        if self.image is None:
            self.status.set_text("Open an image to start.")
            self.fig.canvas.draw_idle()
            return

        self._clear_artists()

        for label in self.labels:
            color = self.colors[label]
            pts = self.points[label]
            if pts:
                xs, ys = zip(*pts)
                artist = self.ax.scatter(xs, ys, s=28, c=color, marker="x", linewidths=1.8)
                self.point_artists[label] = artist

            ellipse = self.ellipses[label]
            if ellipse is not None:
                line_width = 2.8 if label == self.active_label else 1.8
                patch = Ellipse(
                    (ellipse.center_x, ellipse.center_y),
                    width=ellipse.width,
                    height=ellipse.height,
                    angle=ellipse.angle,
                    fill=False,
                    edgecolor=color,
                    linewidth=line_width,
                )
                self.ax.add_patch(patch)
                self.patches[label] = patch

                handle_positions = ellipse.handle_positions()
                handle_artists = []
                for handle_name, pos in handle_positions.items():
                    marker = "o" if handle_name == "center" else "s"
                    artist = self.ax.scatter(
                        [pos[0]],
                        [pos[1]],
                        s=42,
                        c=color,
                        marker=marker,
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=5,
                    )
                    handle_artists.append(artist)
                self.handle_artists[label] = handle_artists

        active_points = len(self.points[self.active_label])
        if self.ellipses[self.active_label] is None:
            if active_points >= 5:
                action = "fit failed; press Z to undo or R to reset"
            else:
                action = f"click {5 - active_points} more point(s)"
        else:
            action = "drag center or square handles to adjust"
        ready = ", ".join([name for name in self.labels if self.ellipses[name] is not None]) or "none"
        batch = ""
        if self.image_paths:
            batch = f" | image {self.image_index + 1}/{len(self.image_paths)}"
        self.status.set_text(f"Active: {self.active_label} | fitted: {ready}{batch} | {action}")
        self.fig.canvas.draw_idle()

    def _annotation_rows(self):
        rows = []
        for label in self.labels:
            ellipse = self.ellipses[label]
            if ellipse is None:
                continue
            row = {"label": label}
            row.update({key: f"{value:.3f}" for key, value in asdict(ellipse).items()})
            rows.append(row)
        return rows

    def save_outputs(self, _event=None):
        if self.image is None or self.image_path is None:
            self.status.set_text("No image loaded.")
            self.fig.canvas.draw_idle()
            return False
        rows = self._annotation_rows()
        if not rows:
            self.status.set_text("Nothing to save yet.")
            self.fig.canvas.draw_idle()
            return False

        out_dir = self.out_dir or (self.image_path.parent / "manual_ellipse_annotations")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        csv_path = out_dir / f"{stem}_ellipses.csv"
        json_path = out_dir / f"{stem}_ellipses.json"
        png_path = out_dir / f"{stem}_ellipses.png"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["label", "center_x", "center_y", "width", "height", "angle"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "image": str(self.image_path),
            "ellipses": {
                label: asdict(ellipse)
                for label, ellipse in self.ellipses.items()
                if ellipse is not None
            },
            "points": self.points,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        annotated = self._render_annotated_image()
        write_image(png_path, annotated)
        self.status.set_text(f"Saved: {csv_path} | {json_path} | {png_path}")
        self.fig.canvas.draw_idle()
        print(f"Saved CSV: {csv_path}")
        print(f"Saved JSON: {json_path}")
        print(f"Saved image: {png_path}")
        return True

    def save_and_next(self, _event=None):
        saved = self.save_outputs()
        if saved and self.image_paths and self.image_index < len(self.image_paths) - 1:
            self.next_image()

    def _render_annotated_image(self):
        annotated = self.image.copy()
        for label in self.labels:
            ellipse = self.ellipses[label]
            if ellipse is None:
                continue
            color_rgb = np.array([255, 204, 0], dtype=np.uint8) if label == "iris" else np.array([0, 230, 118], dtype=np.uint8)
            draw_color = tuple(int(v) for v in color_rgb)
            center = (int(round(ellipse.center_x)), int(round(ellipse.center_y)))
            axes = (int(round(ellipse.width / 2.0)), int(round(ellipse.height / 2.0)))
            cv2.ellipse(annotated, center, axes, ellipse.angle, 0, 360, draw_color, 2, cv2.LINE_AA)
            cv2.circle(annotated, center, 3, draw_color, -1, cv2.LINE_AA)
            text = f"{label}: w={ellipse.width:.1f}, h={ellipse.height:.1f}"
            cv2.putText(
                annotated,
                text,
                (max(5, center[0] - axes[0]), max(20, center[1] - axes[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def quit_app(self, _event=None):
        if self.closing:
            return
        self.closing = True
        plt.close("all")
        sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Manually fit iris and pupil ellipses from 5 clicked points each.")
    parser.add_argument("images", nargs="*", help="Optional image paths. If omitted, a file picker opens.")
    parser.add_argument("--out-dir", default=None, help="Directory for CSV/JSON/annotated PNG outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = ManualEllipseAnnotator(args.images, args.out_dir)
    try:
        plt.show()
    except SystemExit:
        raise
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()

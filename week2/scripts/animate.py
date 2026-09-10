"""Stream rasterized periodic-fluid frames to FFmpeg; no GUI or plotting library."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

WIDTH, HEIGHT = 800, 760

# Small bitmap alphabet for a portable timestamp overlay, independent of fonts.
FONT = {
    "0": (7, 5, 5, 5, 7), "1": (2, 6, 2, 2, 7), "2": (7, 1, 7, 4, 7),
    "3": (7, 1, 7, 1, 7), "4": (5, 5, 7, 1, 1), "5": (7, 4, 7, 1, 7),
    "6": (7, 4, 7, 5, 7), "7": (7, 1, 1, 1, 1), "8": (7, 5, 7, 5, 7),
    "9": (7, 5, 7, 1, 7), ".": (0, 0, 0, 0, 2), "=": (0, 7, 0, 7, 0),
    "T": (7, 2, 2, 2, 2), "S": (7, 4, 7, 1, 7), "E": (7, 4, 7, 4, 7),
    "P": (7, 5, 7, 4, 4), "N": (5, 7, 7, 7, 5), " ": (0, 0, 0, 0, 0),
}


def label(pixels, text, x, y, scale=3):
    for char in text:
        for row, bits in enumerate(FONT[char]):
            for col in range(3):
                if bits & (1 << (2 - col)):
                    for dy in range(scale):
                        start = ((y + row * scale + dy) * WIDTH + x + col * scale) * 3
                        pixels[start:start + scale * 3] = bytes((224, 232, 245)) * scale
        x += 4 * scale


def rasterize(rows, lengths, step, dt):
    pixels = bytearray(bytes((17, 24, 39)) * (WIDTH * HEIGHT))
    scale = min(720 / lengths[0], 640 / lengths[1])
    box_w, box_h = round(lengths[0] * scale), round(lengths[1] * scale)
    left, top = (WIDTH - box_w) // 2, 65
    for y in range(top, top + box_h + 1):
        for x in (left, left + box_w):
            start = (y * WIDTH + x) * 3
            pixels[start:start + 3] = bytes((120, 140, 170))
    for y in (top, top + box_h):
        start = (y * WIDTH + left) * 3
        pixels[start:start + (box_w + 1) * 3] = bytes((120, 140, 170)) * (box_w + 1)
    radius = max(2, round(0.22 * scale))
    for row in rows:
        x, y, vx, vy = (float(row[key]) for key in ("x", "y", "vx", "vy"))
        speed_fraction = min(math.hypot(vx, vy) / 3, 1)
        color = bytes((int(60 + 180 * speed_fraction), int(195 - 65 * speed_fraction), 230))
        # Draw clipped periodic copies so particles cross the boundary smoothly.
        for dx in (-lengths[0], 0, lengths[0]):
            for dy in (-lengths[1], 0, lengths[1]):
                cx = left + round((x + dx) * scale)
                cy = top + box_h - round((y + dy) * scale)
                for py in range(max(top + 1, cy - radius), min(top + box_h, cy + radius + 1)):
                    half = math.isqrt(max(0, radius * radius - (py - cy) ** 2))
                    x0, x1 = max(left + 1, cx - half), min(left + box_w, cx + half + 1)
                    if x1 > x0:
                        start = (py * WIDTH + x0) * 3
                        pixels[start:start + (x1 - x0) * 3] = color * (x1 - x0)
    label(pixels, f"N={len(rows)}  STEP={step}  T={step * dt:.3f}", 40, 25)
    return pixels


def ffmpeg_executable():
    if os.environ.get("IMAGEIO_FFMPEG_EXE"):
        return os.environ["IMAGEIO_FFMPEG_EXE"]
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("install requirements.txt or put ffmpeg on PATH") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def animate(input_dir, output, fps):
    if output.suffix.lower() != ".mp4":
        raise ValueError("video output must use the .mp4 extension")
    meta = json.loads((input_dir / "metadata.json").read_text())
    if meta["format_version"] != 1 or not meta["complete"]:
        raise ValueError("unsupported or incomplete run")
    config = meta["config"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Rename only a successful encode; failures preserve any previous video.
    handle, temporary = tempfile.mkstemp(suffix=".mp4", prefix=".md-video-", dir=output.parent)
    os.close(handle)
    try:
        command = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
                   "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                with (input_dir / "trajectory.csv").open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    for step in range(config["production_steps"] + 1):
                        if step % config["sample_every"] and step != config["production_steps"]:
                            continue
                        rows = []
                        for particle in range(config["particles"]):
                            row = next(reader)
                            if int(row["step"]) != step or int(row["id"]) != particle:
                                raise ValueError("invalid trajectory ordering")
                            rows.append(row)
                        process.stdin.write(rasterize(rows, meta["box_lengths"], step, config["dt"]))
                    if next(reader, None) is not None:
                        raise ValueError("extra trajectory rows")
                process.stdin.close()
                status = process.wait()
                if status:
                    errors.seek(0)
                    raise RuntimeError(f"FFmpeg exited {status}: {errors.read().decode(errors='replace')}")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.fps <= 240:
        parser.error("fps must be between 1 and 240")
    try:
        animate(args.input, args.output, args.fps)
    except (OSError, ValueError, RuntimeError, StopIteration, KeyError) as error:
        print(f"video: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Exercise the real md CLI, including malformed recordings and video decoding."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def binary():
    supplied = os.environ.get("MD_BINARY")
    if supplied:
        return Path(supplied).resolve()
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(ROOT / "md/Cargo.toml")], check=True)
    return ROOT / "md/target/release/md"


def invoke(binary, *args):
    return subprocess.run([str(binary), *map(str, args)], text=True, capture_output=True)


@pytest.fixture
def short_run(binary, tmp_path):
    output = tmp_path / "run"
    result = invoke(binary, "run", "--output", output, "--equilibration-steps", 20,
                    "--steps", 113, "--sample-every", 50)
    assert result.returncode == 0, result.stderr
    return output


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_recording_contract_and_reproducibility(binary, short_run, tmp_path):
    metadata = json.loads((short_run / "metadata.json").read_text())
    assert metadata["format_version"] == 1 and metadata["complete"] is True
    assert metadata["config"]["particles"] == 100
    assert metadata["config"]["seed"] == 2026
    assert math.isclose(math.prod(metadata["box_lengths"]), 125.0)
    rows = read_csv(short_run / "observables.csv")
    assert len(rows) == 114
    for step, row in enumerate(rows):
        assert int(row["step"]) == step
        assert math.isclose(float(row["time"]), step * 0.005, abs_tol=1e-15)
        assert all(math.isfinite(float(value)) for value in row.values())
        assert math.isclose(float(row["total"]), float(row["kinetic"]) + float(row["potential"]))
        assert math.isclose(float(row["temperature"]), float(row["kinetic"]) / 99)
    frames = read_csv(short_run / "trajectory.csv")
    assert len(frames) == 400
    assert sorted({int(row["step"]) for row in frames}) == [0, 50, 100, 113]
    for offset in range(0, len(frames), 100):
        assert [int(row["id"]) for row in frames[offset:offset + 100]] == list(range(100))
    for row in frames:
        assert 0 <= float(row["x"]) < metadata["box_lengths"][0]
        assert 0 <= float(row["y"]) < metadata["box_lengths"][1]
    second = tmp_path / "second"
    result = invoke(binary, "run", "--output", second, "--equilibration-steps", 20,
                    "--steps", 113, "--sample-every", 50)
    assert result.returncode == 0, result.stderr
    for name in ("metadata.json", "observables.csv", "trajectory.csv"):
        assert (short_run / name).read_bytes() == (second / name).read_bytes()
    assert invoke(binary, "run", "--output", short_run).returncode != 0


@pytest.mark.parametrize("args", [
    ("--dt", "0"), ("--dt", "NaN"), ("--density", "0"),
    ("--particles", "99"), ("--cutoff", "8"), ("--sample-every", "0"),
    ("--temperature", "0"), ("--steps", "0"), ("--unknown", "2"),
])
def test_invalid_configuration(binary, tmp_path, args):
    output = tmp_path / "invalid"
    result = invoke(binary, "run", "--output", output, *args)
    assert result.returncode != 0 and result.stderr
    assert not output.exists()


def test_help(binary):
    result = invoke(binary, "--help")
    assert result.returncode == 0
    assert all(command in result.stdout for command in ("run", "check", "video", "dimer"))


@pytest.mark.parametrize("damage", ["truncated", "nan", "wrong_id", "metadata", "wrong_energy"])
def test_checker_rejects_malformed_run(binary, short_run, damage):
    path = short_run / "trajectory.csv"
    lines = path.read_text().splitlines()
    if damage == "truncated":
        path.write_text("\n".join(lines[:-1]) + "\n")
    elif damage == "nan":
        fields = lines[1].split(",")
        fields[2] = "NaN"
        lines[1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n")
    elif damage == "wrong_id":
        fields = lines[1].split(",")
        fields[1] = "1"
        lines[1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n")
    elif damage == "wrong_energy":
        path = short_run / "observables.csv"
        lines = path.read_text().splitlines()
        fields = lines[1].split(",")
        fields[4] = "1234"
        lines[1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n")
    else:
        path = short_run / "metadata.json"
        meta = json.loads(path.read_text())
        meta["complete"] = False
        path.write_text(json.dumps(meta))
    result = invoke(binary, "check", "--input", short_run)
    assert result.returncode != 0
    assert result.stderr and "PASS" not in result.stdout


def test_checker_failure_exit_code(binary, tmp_path):
    # A cold lattice with no equilibration transfers kinetic energy to the potential.
    output = tmp_path / "unequilibrated"
    result = invoke(binary, "run", "--output", output, "--equilibration-steps", 0, "--steps", 500)
    assert result.returncode == 0, result.stderr
    report = invoke(binary, "check", "--input", output)
    assert report.returncode == 1
    assert "temperature: FAIL" in report.stdout


def test_video_is_decodable_and_has_all_frames(binary, short_run, tmp_path):
    import imageio_ffmpeg
    output = tmp_path / "fluid.mp4"
    result = invoke(binary, "video", "--input", short_run, "--output", output, "--python", sys.executable)
    assert result.returncode == 0, result.stderr
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    metadata = next(reader)
    assert metadata["size"] == (800, 760)
    frames = list(reader)
    assert len(frames) == 4
    assert frames[0] != frames[-1]


def test_video_propagates_encoder_failure(binary, short_run, tmp_path):
    env = dict(os.environ, IMAGEIO_FFMPEG_EXE=str(tmp_path / "missing-ffmpeg"))
    result = subprocess.run([str(binary), "video", "--input", str(short_run),
                             "--output", str(tmp_path / "bad.mp4"), "--python", sys.executable],
                            text=True, capture_output=True, env=env)
    assert result.returncode != 0

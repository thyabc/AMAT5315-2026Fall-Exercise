"""Check the actual Rust executable's CSV contract with pytest."""

import csv
import io
import math
from pathlib import Path
import subprocess


def test_dimer_csv():
    manifest = Path(__file__).parent / "md" / "Cargo.toml"
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(manifest)],
        check=True, capture_output=True, text=True,
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    assert reader.fieldnames == [
        "integrator", "step", "time", "kinetic", "potential", "total", "relative_error"
    ]
    rows = list(reader)
    assert len(rows) == 1002
    assert {row["integrator"] for row in rows} == {"euler", "velocity-verlet"}
    for method in ("euler", "velocity-verlet"):
        samples = [row for row in rows if row["integrator"] == method]
        assert len(samples) == 501
        initial_energy = float(samples[0]["total"])
        assert initial_energy < 0.0
        assert float(samples[0]["kinetic"]) == 0.0
        for step, row in enumerate(samples):
            assert int(row["step"]) == step
            assert float(row["time"]) == step * 0.01
            values = {key: float(row[key]) for key in (
                "kinetic", "potential", "total", "relative_error"
            )}
            assert all(math.isfinite(value) for value in values.values())
            assert math.isclose(values["total"], values["kinetic"] + values["potential"])
            expected = (values["total"] - initial_energy) / abs(initial_energy)
            assert math.isclose(values["relative_error"], expected, abs_tol=1e-14)

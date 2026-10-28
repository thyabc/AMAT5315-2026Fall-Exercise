"""Self-contained HTML/SVG evidence views. Only the Python standard library."""
import argparse
import csv
from html import escape
import json
import math
from pathlib import Path

COLORS = ("#176b91", "#b84326", "#427a35", "#79539b")


def load_artifacts(directory):
    directory = Path(directory)
    data = {}
    for name in ("baselines", "causality", "eval", "config", "resume"):
        path = directory / f"{name}.json"
        if path.exists():
            data[name] = json.loads(path.read_text())
    path = directory / "train_log.csv"
    if path.exists():
        with path.open() as stream:
            data["log"] = [{k: float(v) for k, v in row.items() if v}
                           for row in csv.DictReader(stream)]
    data["comparison"] = [{"name": path.name, **load_artifacts(path)}
                          for path in sorted((directory / "comparison").glob("*/"))
                          if (path / "eval.json").exists()]
    return data


def table(headers, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{escape(str(x))}</th>" for x in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(
                f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
            + "</tbody></table>")


def label(x, y, value, **attrs):
    attributes = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{escape(str(value))}</text>'


def svg(body, title):
    return (f'<svg viewBox="0 0 800 420" role="img" xmlns="http://www.w3.org/2000/svg">'
            f'<title>{escape(title)}</title><rect width="800" height="420" fill="white"/>'
            '<g font-family="sans-serif" font-size="13">' + body + '</g></svg>')


def lines(series, xlabel="Training step", ylabel="Cross-entropy [nats/character]"):
    """Series are (label, x values, y values); show full zero-based y range."""
    points = [(x, y) for _, xs, ys in series for x, y in zip(xs, ys)]
    if not points or not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError("Plot requires finite, nonempty data")
    xmax = max(1, max(x for x, _ in points))
    ymax = max(1e-9, max(y for _, y in points)) * 1.08
    body = ""
    for i in range(6):
        y = 345 - i * 56
        body += f'<path d="M85 {y}H765" stroke="#ddd"/>'
        body += label(75, y + 4, f"{ymax * i / 5:.3g}", text_anchor="end")
        body += label(85 + i * 136, 368, f"{xmax * i / 5:.4g}", text_anchor="middle")
    body += label(425, 405, xlabel, text_anchor="middle")
    body += '<text transform="translate(19,210) rotate(-90)" text-anchor="middle">' + escape(ylabel) + '</text>'
    for index, (name, xs, ys) in enumerate(series):
        if len(xs) != len(ys):
            raise ValueError("Plot coordinates differ in length")
        color = COLORS[index % len(COLORS)]
        coords = " ".join(f"{85 + x / xmax * 680:.2f},{345 - y / ymax * 280:.2f}"
                          for x, y in zip(xs, ys))
        body += f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        for x, y in zip(xs, ys):
            body += f'<circle cx="{85+x/xmax*680:.2f}" cy="{345-y/ymax*280:.2f}" r="2.5" fill="{color}"><title>{escape(name)}: {x:g}, {y:.6g}</title></circle>'
        body += label(85 + index * 210, 27, name, fill=color)
    return svg(body, f"{ylabel} versus {xlabel}")


def heatmap(attention, text):
    n = len(attention)
    if not n or any(len(row) != n for row in attention):
        raise ValueError("Attention must be a nonempty square matrix")
    size = 285 / n
    body = label(365, 20, "Attention probability, layer 0 / head 0", text_anchor="middle")
    for i, row in enumerate(attention):
        for j, weight in enumerate(row):
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("Invalid attention probability")
            color = f"rgb({round(245-220*weight)},{round(250-145*weight)},{round(255-110*weight)})"
            body += (f'<rect x="{225+j*size:.2f}" y="{45+i*size:.2f}" width="{size:.2f}" height="{size:.2f}" fill="{color}" stroke="#ddd" stroke-width=".3">'
                     f'<title>query {i}, key {j}: {weight:.6g}</title></rect>')
        char = repr(text[i])[1:-1] if i < len(text) else ""
        body += label(215, 45+(i+.65)*size, f"{i}: {char}", text_anchor="end", font_size=min(13, size*.7))
        body += label(225+(i+.5)*size, 349, i, text_anchor="middle", font_size=min(13, size*.7))
    body += label(365, 377, "Key position (character attended to)", text_anchor="middle")
    body += '<text transform="translate(100,190) rotate(-90)" text-anchor="middle">Query position</text>'
    body += label(545, 120, "White = 0") + label(545, 145, "Dark blue = 1")
    return svg(body, "Causal attention probabilities")


def score_rows(data):
    return [(name.capitalize(), f'{data[name]["loss"]:.6f}', f'{data[name]["ppl"]:.6f}')
            for name in ("uniform", "frequency")]


def write_page(path, title, body, source):
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    f'<title>{escape(title)}</title><style>body{{font:17px/1.5 system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;color:#192833}}'
                    'svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{text-align:left;padding:.4rem;border-bottom:1px solid #ddd}'
                    'pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style>'
                    f'<h1>{escape(title)}</h1>{body}<details><summary>Source measurements</summary>'
                    f'<pre>{escape(json.dumps(source, indent=2))}</pre></details></html>')


def render_all(directory):
    directory = Path(directory)
    data = load_artifacts(directory)
    if "baselines" in data:
        b = data["baselines"]
        probabilities = b["train_probabilities"]
        body = lines([("Training-frequency prediction", list(range(len(probabilities))), probabilities)],
                     "Character ID (sorted vocabulary)", "Probability")
        body += table(("Predictor", "Loss [nats/character]", "Character PPL"), score_rows(b))
        body += '<p>Frequency probabilities use training counts with add-one smoothing. Scores use the fixed validation targets.</p><pre>' + escape(repr(b["vocabulary"])) + '</pre>'
        write_page(directory / "baselines.html", "Text as probability", body, b)
    if "causality" in data:
        c = data["causality"]
        ranked = sorted(zip(c["vocabulary"], c["next_probabilities"]), key=lambda pair: -pair[1])
        body = '<pre>Input: ' + escape(repr(c["input"])) + '</pre>' + heatmap(c["attention"], c["input"])
        identity = (f'Checkpoint {c["checkpoint_sha256"]}' if c.get("checkpoint_sha256")
                    else f'Untrained initialization, seed {c["initialization_seed"]}')
        body += '<p>' + escape(identity) + '</p>'
        body += '<h2>Next character after the complete input</h2>'
        body += table(("Character", "Probability"), [(repr(char), f"{p:.6f}") for char, p in ranked[:12]])
        body += f'<p>Top 12 shown; remaining probability {sum(p for _, p in ranked[12:]):.6f}. Maximum prefix error: {c["prefix_error"]:.6g}.</p>'
        write_page(directory / "attention.html", "A model that sees only the past", body, c)
    if "log" in data:
        log = data["log"]
        body = lines([(name, [r["step"] for r in log], [r[key] for r in log])
                      for name, key in (("Training minibatch", "train_loss"), ("Validation", "val_loss"))])
        if "eval" in data:
            e = data["eval"]
            body += f'<p>Saved-weight evaluation: {e["loss"]:.6f} nats/character; PPL {e["ppl"]:.6f}.</p>'
        body += '<p>Training values are minibatch losses; validation uses the fixed evaluation policy. PPL = exp(cross-entropy).</p>'
        write_page(directory / "loss.html", "Training and held-out prediction", body, data)
    if data["comparison"]:
        runs = data["comparison"]
        body = lines([(r["name"], [v["tokens"] for v in r["log"]], [v["val_loss"] for v in r["log"]]) for r in runs], "Training tokens")
        body += table(("Run", "Width", "Tokens", "Wall time [s]", "Loss", "PPL"),
                      [(r["name"], r["config"]["n_embd"], int(r["log"][-1]["tokens"]),
                        f'{r["config"]["elapsed_seconds"]:.2f}', f'{r["eval"]["loss"]:.6f}', f'{r["eval"]["ppl"]:.6f}') for r in runs])
        body += table(("Run", "Device", "Precision"), [(r["name"], r["config"].get("device_kind", r["config"]["device"]), r["config"]["precision"]) for r in runs])
        body += '<p>Compare the same token budget, context policy, data order, schedule, and device. A single pair of seeds does not establish an average improvement.</p>'
        write_page(directory / "comparison.html", "Change one thing", body, runs)
    if "resume" in data:
        r = data["resume"]
        body = lines([(name, r["steps"], r[key]) for name, key in
                      (("Uninterrupted", "uninterrupted_losses"), ("Fresh-process resume", "resumed_losses"))])
        body += table(("Continuation residual", "Maximum absolute error"),
                      [(key, f'{r[key]:.8g}') for key in ("max_loss_error", "max_parameter_error", "max_optimizer_error")])
        body += '<p>Overlapping traces agree. The saved optimizer and random state are checked alongside the parameters on the same backend.</p>'
        write_page(directory / "resume.html", "Training across interruption", body, r)


def self_check():
    """A known two-symbol distribution also exercises HTML escaping and SVG parsing."""
    import tempfile
    import xml.etree.ElementTree as ET

    assert ET.fromstring(lines([("loss", [0, 1], [math.log(2), 0])])).tag.endswith("svg")
    assert ET.fromstring(heatmap([[1, 0], [.25, .75]], "<>")).tag.endswith("svg")
    try:
        lines([("bad", [0], [float("nan")])])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite data must fail")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        score = {"loss": math.log(2), "ppl": 2}
        (root / "baselines.json").write_text(json.dumps({"vocabulary": ["<", ">"],
            "train_probabilities": [.5, .5], "uniform": score, "frequency": score}))
        (root / "causality.json").write_text(json.dumps({"input": "<>",
            "attention": [[1, 0], [.25, .75]], "vocabulary": ["<", ">"],
            "next_probabilities": [.5, .5], "prefix_error": 0,
            "checkpoint_sha256": None, "initialization_seed": 7}))
        render_all(root)
        page = (root / "baselines.html").read_text()
        assert "2.000000" in page and "&lt;" in page and "Source measurements" in page
        attention_page = (root / "attention.html").read_text()
        assert "Untrained initialization, seed 7" in attention_page
        assert "Checkpoint None" not in attention_page
    print("plots self-check PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.artifacts:
        render_all(args.artifacts)
    else:
        parser.error("provide an artifact directory or --self-check")

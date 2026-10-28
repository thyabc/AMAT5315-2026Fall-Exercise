import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm

root = Path(__file__).resolve().parents[1]
out = root / "cpu/work/part2"
config = json.loads((root / "cpu/configs/part2.json").read_text())

# Load existing model: NO training
params, meta = lm.load(out)
independent = json.loads((out / "independent-eval.json").read_text())

# Recover actual training curve from the saved log
log = (root / "artifacts/part2-training.log").read_text()
matches = re.findall(
    r"step\s+(\d+): train=([\d.]+)\s+val\[96\]=([\d.]+)\s+wall=(\d+)s",
    log
)
rows = [
    dict(step=int(s), train_loss=float(t),
         val_loss_subset=float(v), wall_seconds=int(w))
    for s, t, v, w in matches
]
assert rows and rows[-1]["step"] == 3000, "Training log incomplete"

# Check that future characters cannot affect earlier logits
probe = lm.causality_probe(params, config)

# Inspect attention of all four heads in layer 1
prompt = "ROMEO:\nBut soft, what light"
vocab = lm.corpus()[2]["vocabulary"]
ids = np.array([[vocab.index(c) for c in prompt]], dtype=np.int32)
_, attention = lm.forward(params, ids, config, return_attention=True)
attention = np.asarray(attention)[0]
upper = float(np.max(np.abs(np.triu(attention, k=1))))
row_error = float(np.max(np.abs(attention.sum(-1) - 1)))

# Generate 500 new characters at each temperature
samples = []
for temperature in (0.5, 1.0):
    generated = lm.generate(
        params, config, vocab, "ROMEO:\n",
        temperature, 500, 7
    )
    samples.append(dict(
        temperature=temperature, seed=7,
        length=500, text=generated
    ))

# Save data needed for the figures
lm.save_arrays(
    part2_attention=attention,
    part2_prompt=np.frombuffer(prompt.encode(), dtype=np.uint8),
    part2_steps=np.array([r["step"] for r in rows]),
    part2_train=np.array([r["train_loss"] for r in rows]),
    part2_val=np.array([r["val_loss_subset"] for r in rows]),
)

lm.record("part2", dict(
    config=config,
    parameters=lm.parameter_counts(config),
    curve=rows,
    independent_score=independent,
    causality=probe,
    attention_prompt=prompt,
    attention_upper_triangle_max=upper,
    attention_row_sum_max_error=row_error,
    samples=samples,
    checkpoint_sha256=meta["checkpoint_sha256"],
))

(root / "artifacts/part2-samples.json").write_text(
    json.dumps(samples, indent=2) + "\n"
)

print("FULL LOSS:", independent["loss"])
print("PPL:", independent["ppl"])
print("CAUSALITY:", probe)
print("FUTURE ATTENTION MAX:", upper)
print("ATTENTION ROW ERROR:", row_error)
print("TRAINING LOG POINTS:", len(rows))
print("GENERATED CHARACTERS:", [len(s["text"]) - len("ROMEO:\n") for s in samples])
print("Part 2 evidence data saved successfully.")

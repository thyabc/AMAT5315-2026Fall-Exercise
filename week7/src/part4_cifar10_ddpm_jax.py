"""AMAT5315 Week 7 Part 4: pure-JAX DDPM sampler for a frozen CIFAR-10 UNet2DModel.

Architecture mapped from Hugging Face diffusers v0.39.0 (UNet2DModel,
ResnetBlock2D, AttnDownBlock2D, AttnUpBlock2D, UNetMidBlock2D).

No FlaxUNet2DModel is imported: that class does not exist in diffusers.
The checkpoint is only read by safetensors; forward computation uses JAX.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "inputs" / "cifar-model"
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"

STEPS = 1000
BETA_START, BETA_END = 0.0001, 0.02
SEED = 20261021


def conv2d(x, p, base, stride=1, padding=1):
    w = p[base + ".weight"]
    b = p[base + ".bias"]
    y = jax.lax.conv_general_dilated(
        x, w, (stride, stride), ((padding, padding), (padding, padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + b[None, :, None, None]


def linear(x, p, base):
    w = p[base + ".weight"]
    if w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]  # legacy 1x1 Conv1d attention projection
    if w.ndim == 4 and w.shape[-2:] == (1, 1):
        w = w[..., 0, 0]
    b = p.get(base + ".bias")
    y = jnp.matmul(x, jnp.swapaxes(w, -1, -2))
    return y if b is None else y + b


def group_norm(x, p, base, groups=32, eps=1e-6):
    batch, channels, height, width = x.shape
    xg = jnp.reshape(x, (batch, groups, channels // groups, height, width))
    mu = jnp.mean(xg, axis=(2, 3, 4), keepdims=True)
    var = jnp.mean((xg - mu)**2, axis=(2, 3, 4), keepdims=True)
    out = jnp.reshape((xg - mu) * jax.lax.rsqrt(var + eps), x.shape)
    return (out * p[base + ".weight"][None, :, None, None]
            + p[base + ".bias"][None, :, None, None])


def timestep_embedding(t, dim, freq_shift=1, flip_sin_to_cos=False):
    half = dim // 2
    freq = jnp.exp(-np.log(10000.0) *
                   jnp.arange(half, dtype=jnp.float32) / (half - freq_shift))
    args = jnp.asarray(t, dtype=jnp.float32).reshape(-1, 1) * freq[None, :]
    sine, cosine = jnp.sin(args), jnp.cos(args)
    out = jnp.concatenate((cosine, sine) if flip_sin_to_cos else
                          (sine, cosine), axis=-1)
    if dim % 2:
        out = jnp.pad(out, ((0, 0), (0, 1)))
    return out


def resnet(x, emb, p, prefix, cfg):
    groups = cfg["norm_num_groups"]
    eps = cfg["norm_eps"]
    h = group_norm(x, p, prefix + ".norm1", groups, eps)
    h = jax.nn.silu(h)
    h = conv2d(h, p, prefix + ".conv1")
    # Diffusers ResnetBlock2D adds the time projection *before* norm2.
    e = linear(jax.nn.silu(emb), p, prefix + ".time_emb_proj")
    h = h + e[:, :, None, None]
    h = group_norm(h, p, prefix + ".norm2", groups, eps)
    h = conv2d(jax.nn.silu(h), p, prefix + ".conv2")
    if prefix + ".conv_shortcut.weight" in p:
        x = conv2d(x, p, prefix + ".conv_shortcut", padding=0)
    return x + h


def attention(x, p, prefix, cfg):
    original = x
    groups = cfg["norm_num_groups"]
    eps = cfg["norm_eps"]
    normbase = (prefix + ".group_norm" if prefix + ".group_norm.weight" in p
                else prefix + ".norm")
    x = group_norm(x, p, normbase, groups, eps)
    b, c, h, w = x.shape
    seq = jnp.transpose(jnp.reshape(x, (b, c, h * w)), (0, 2, 1))
    legacy = prefix + ".query.weight" in p
    qname = prefix + (".query" if legacy else ".to_q")
    kname = prefix + (".key" if legacy else ".to_k")
    vname = prefix + (".value" if legacy else ".to_v")
    oname = prefix + (".proj_attn" if legacy else ".to_out.0")
    q, k, v = [linear(seq, p, s) for s in (qname, kname, vname)]
    heads = max(1, c // (cfg.get("attention_head_dim") or c))
    head_dim = q.shape[-1] // heads

    def to_heads(y):
        return jnp.transpose(jnp.reshape(y, (b, h * w, heads, head_dim)),
                             (0, 2, 1, 3))

    q, k, v = [to_heads(y) for y in (q, k, v)]
    scores = (q @ jnp.swapaxes(k, -2, -1)) / np.sqrt(head_dim)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    values = weights @ v
    values = jnp.reshape(jnp.transpose(values, (0, 2, 1, 3)),
                         (b, h * w, heads * head_dim))
    values = linear(values, p, oname)
    values = jnp.transpose(values, (0, 2, 1)).reshape((b, c, h, w))
    return original + values


def downsample(x, p, base, cfg):
    pad = cfg.get("downsample_padding", 1)
    if pad == 0:
        # This detail is essential for the provided CIFAR-10 config.
        # diffusers Downsample2D pads bottom/right before stride-2 convolution.
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
    return conv2d(x, p, base, stride=2, padding=pad)


def upsample(x, p, base):
    x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
    return conv2d(x, p, base)


def forward_unet(x, timesteps, p, cfg):
    # x, output: NCHW; all operations and weights are JAX float32.
    t = timestep_embedding(timesteps, cfg["block_out_channels"][0],
                           cfg.get("freq_shift", 0),
                           cfg.get("flip_sin_to_cos", True))
    emb = linear(t, p, "time_embedding.linear_1")
    emb = linear(jax.nn.silu(emb), p, "time_embedding.linear_2")
    x = conv2d(x, p, "conv_in")
    skips = [x]
    down_types = cfg["down_block_types"]
    up_types = cfg["up_block_types"]
    num_layers = cfg["layers_per_block"]

    for stage, typ in enumerate(down_types):
        for idx in range(num_layers):
            prefix = f"down_blocks.{stage}"
            x = resnet(x, emb, p, f"{prefix}.resnets.{idx}", cfg)
            if typ == "AttnDownBlock2D":
                x = attention(x, p, f"{prefix}.attentions.{idx}", cfg)
            skips.append(x)
        if stage != len(down_types) - 1:
            x = downsample(x, p, f"down_blocks.{stage}.downsamplers.0.conv", cfg)
            skips.append(x)

    x = resnet(x, emb, p, "mid_block.resnets.0", cfg)
    if "mid_block.attentions.0.to_q.weight" in p or \
       "mid_block.attentions.0.query.weight" in p:
        x = attention(x, p, "mid_block.attentions.0", cfg)
    x = resnet(x, emb, p, "mid_block.resnets.1", cfg)

    for stage, typ in enumerate(up_types):
        for idx in range(num_layers + 1):
            x = jnp.concatenate((x, skips.pop()), axis=1)
            prefix = f"up_blocks.{stage}"
            x = resnet(x, emb, p, f"{prefix}.resnets.{idx}", cfg)
            if typ == "AttnUpBlock2D":
                x = attention(x, p, f"{prefix}.attentions.{idx}", cfg)
        if stage != len(up_types) - 1:
            x = upsample(x, p, f"up_blocks.{stage}.upsamplers.0.conv")

    assert not skips, f"Unused U-Net skip tensors: {len(skips)}"
    x = group_norm(x, p, "conv_norm_out", cfg["norm_num_groups"], cfg["norm_eps"])
    return conv2d(jax.nn.silu(x), p, "conv_out")


def load_predictor():
    with (MODEL_DIR / "config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("_class_name") != "UNet2DModel":
        raise ValueError("Unexpected architecture; expected UNet2DModel")
    down = cfg["down_block_types"]
    up = cfg["up_block_types"]
    if not set(down).issubset({"DownBlock2D", "AttnDownBlock2D"}) or \
       not set(up).issubset({"UpBlock2D", "AttnUpBlock2D"}):
        raise ValueError("This reference port supports only plain/attention 2D U-Net blocks")
    if cfg.get("act_fn", "silu") != "silu" or cfg.get("sample_size") != 32:
        raise ValueError("Unexpected activation or CIFAR-10 image size")
    raw = load_file(str(MODEL_DIR / "diffusion_pytorch_model.safetensors"))
    if not raw:
        raise ValueError("Checkpoint has no tensors")
    if all(name.startswith("model.") for name in raw):
        raw = {name[6:]: weight for name, weight in raw.items()}
    p = {name: jnp.asarray(weight, dtype=jnp.float32) for name, weight in raw.items()}
    print(f"Read {len(p)} frozen tensors from safetensors (JAX, no PyTorch forward).", flush=True)
    return p, cfg


def to_uint8(images):
    # Display clipping only, never clip the sampler's internal state.
    images = np.transpose(np.asarray(images, np.float32), (0, 2, 3, 1))
    return np.round(np.clip((images + 1) * 127.5, 0, 255)).astype(np.uint8)


def save_montage(images, path, columns=2, upscale=4):
    n, h, w, channels = images.shape
    rows = (n + columns - 1) // columns
    canvas = np.full((rows * h, columns * w, channels), 255, dtype=np.uint8)
    for i, pic in enumerate(images):
        r, c = divmod(i, columns)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = pic
    result = Image.fromarray(canvas)
    result.resize((columns * w * upscale, rows * h * upscale),
                  resample=Image.Resampling.NEAREST).save(path)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--check", action="store_true", help="One forward pass only (no sampling)")
    cli.add_argument("--run", action="store_true", help="Generate four images using 1000 DDPM steps")
    args = cli.parse_args()
    if not (args.check or args.run):
        cli.error("Choose --check first, then --run")

    print("Backend:", jax.default_backend(), flush=True)
    params, cfg = load_predictor()
    def predict_impl(x, timesteps, weights):
        return forward_unet(x, timesteps, weights, cfg)

    forward_jit = jax.jit(predict_impl)

    # The first call compiles. Nothing is trained or modified.
    key = jax.random.PRNGKey(SEED)
    test_x = jax.random.normal(key, (1, 3, 32, 32), dtype=jnp.float32)
    print("Compiling and testing one JAX U-Net evaluation ...", flush=True)
    t0 = time.time()
    prediction = forward_jit(test_x, jnp.array([999], dtype=jnp.float32), params)
    prediction.block_until_ready()
    arr = np.asarray(prediction)
    print(f"Prediction shape={arr.shape}; finite={np.isfinite(arr).all()};"
          f" RMS={np.sqrt(np.mean(arr**2)):.6f}; compile/run {time.time()-t0:.1f}s", flush=True)
    if arr.shape != (1, 3, 32, 32) or not np.isfinite(arr).all():
        raise ValueError("JAX U-Net forward test failed")
    if args.check:
        print("CHECK OK. Next: run --run.")
        return

    EVIDENCE.mkdir(exist_ok=True, parents=True)
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    betas = jnp.linspace(BETA_START, BETA_END, STEPS, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (4, 3, 32, 32), dtype=jnp.float32)
    snapshots = {1000: np.asarray(x)}
    print("Running 1000 DDPM reverse steps on four images...", flush=True)
    started = time.time()
    for t in range(STEPS, 0, -1):
        i = t - 1
        eps = forward_jit(x, jnp.full((4,), i, dtype=jnp.float32), params)
        mu = (x - betas[i] / jnp.sqrt(1.0 - alpha_bars[i]) * eps) / jnp.sqrt(alphas[i])
        if t > 1:
            key, subkey = jax.random.split(key)
            x = mu + jnp.sqrt(betas[i]) * jax.random.normal(subkey, x.shape)
        else:
            x = mu  # final step: no noise, no clipping
        if t in (800, 600, 400, 200, 100, 50, 10, 1):
            x.block_until_ready()
            snapshots[t] = np.asarray(x)
        if t % 100 == 0 or t == 1:
            print(f"  reverse step t={t:4d}; elapsed={time.time()-started:.1f}s", flush=True)

    final = np.asarray(x)
    final_display = to_uint8(final)
    save_montage(final_display, EVIDENCE / "part4_cifar10_generated_grid.png")
    for idx, image in enumerate(final_display):
        Image.fromarray(image).save(EVIDENCE / f"part4_cifar10_sample_{idx+1}.png")
    states = sorted(snapshots.keys(), reverse=True)
    # Each row follows one image over time.
    strip = np.concatenate([to_uint8(snapshots[s]) for s in states], axis=2)
    Image.fromarray(np.concatenate(strip, axis=0)).resize(
        (len(states)*32*2, 4*32*2), Image.Resampling.NEAREST
    ).save(EVIDENCE / "part4_cifar10_trajectory.png")
    np.savez_compressed(ARTIFACTS / "part4_cifar10_samples.npz", samples=final,
                        levels=np.array(states), states=np.stack([snapshots[s] for s in states]))
    settings = dict(seed=SEED, n_images=4, diffusion_steps=STEPS,
                    beta_start=BETA_START, beta_end=BETA_END,
                    model="google/ddpm-cifar10-32", sampler="DDPM",
                    framework="JAX", model_index="t-1", reverse_variance="beta_t",
                    no_clipping=True, final_noise=False)
    (ARTIFACTS / "part4_cifar10_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")
    print("PART 4 FINISHED. Check evidence/part4_cifar10_generated_grid.png", flush=True)


if __name__ == "__main__":
    main()

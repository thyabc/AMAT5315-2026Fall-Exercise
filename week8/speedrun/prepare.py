"""Pinned TinyShakespeare, sorted character vocabulary and contiguous 90/10 split."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

CORPUS_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
CORPUS_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'


def prepare(directory='data'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'input.txt'
    if not source.exists():
        cached = Path(__file__).with_name('input.txt')
        source.write_bytes(cached.read_bytes() if cached.exists() else
                           urllib.request.urlopen(CORPUS_URL, timeout=60).read())
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ValueError('TinyShakespeare SHA-256 mismatch')
    text = raw.decode('utf-8')
    chars = sorted(set(text))
    ids = np.array([chars.index(ch) for ch in text], dtype=np.uint16)
    split = int(0.9 * len(ids))
    ids[:split].tofile(directory / 'train.bin')
    ids[split:].tofile(directory / 'val.bin')
    meta = dict(corpus_sha256=CORPUS_SHA256, vocabulary=chars, vocab_size=len(chars),
                train_tokens=split, val_tokens=len(ids)-split, train_frac=0.9)
    (directory / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return ids[:split], ids[split:], meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    print(prepare(parser.parse_args().data_dir)[2])

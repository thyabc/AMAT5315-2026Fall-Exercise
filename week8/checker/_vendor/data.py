"""Checker-owned pinned corpus. Evaluation never downloads or trusts student data."""
import hashlib
from pathlib import Path

CORPUS_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'


def _load_corpus_bytes():
    source = Path(__file__).resolve().parents[2] / 'speedrun' / 'input.txt'
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ValueError('course corpus SHA-256 mismatch')
    return raw

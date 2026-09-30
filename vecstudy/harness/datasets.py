"""harness/datasets.py — dataset loaders, synthetic generator, ground truth."""

from __future__ import annotations

import hashlib
import struct
import urllib.request
import tarfile
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
SIFT_FTP_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_EXPECTED_BYTES = 168_280_445


def read_fvecs(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    d = struct.unpack_from("<i", data, 0)[0]
    record_size = 4 + d * 4
    n = len(data) // record_size
    arr = np.frombuffer(data, dtype=np.float32).reshape(n, d + 1)
    return np.ascontiguousarray(arr[:, 1:])


def read_bvecs(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    d = struct.unpack_from("<i", data, 0)[0]
    record_size = 4 + d
    n = len(data) // record_size
    out = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        out[i] = np.frombuffer(data, dtype=np.uint8, count=d,
                               offset=i * record_size + 4).astype(np.float32)
    return out


def read_ivecs(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    d = struct.unpack_from("<i", data, 0)[0]
    record_size = 4 + d * 4
    n = len(data) // record_size
    arr = np.frombuffer(data, dtype=np.int32).reshape(n, d + 1)
    return np.ascontiguousarray(arr[:, 1:])


def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (X / norms).astype(np.float32)


def fetch_sift(data_root: Path | None = None) -> Path:
    root = data_root or DATA_ROOT
    sift_dir = root / "sift"
    if (sift_dir / "sift_base.fvecs").exists():
        return sift_dir
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "sift.tar.gz"
    if not tar_path.exists():
        print(f"Downloading SIFT1M (~168 MB) …")
        urllib.request.urlretrieve(SIFT_FTP_URL, str(tar_path), _progress_hook)
        print()
    actual = tar_path.stat().st_size
    if actual != SIFT_EXPECTED_BYTES:
        raise RuntimeError(f"sift.tar.gz size mismatch: expected {SIFT_EXPECTED_BYTES}, got {actual}")
    print(f"Extracting …")
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(root), filter="data")
    print("Done.")
    return sift_dir


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  {pct:3d}%  {downloaded/1_048_576:.1f} MB", end="", flush=True)


def make_synthetic(n: int = 10_000, dim: int = 128,
                   n_topics: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    topic_centres = rng.randn(n_topics, dim).astype(np.float32)
    topic_centres /= np.linalg.norm(topic_centres, axis=1, keepdims=True)
    spectrum = (1.0 / np.sqrt(np.arange(1, dim + 1))).astype(np.float32)
    assignments = rng.randint(0, n_topics, size=n)
    X = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        noise = rng.randn(dim).astype(np.float32) * spectrum
        X[i] = topic_centres[assignments[i]] + 0.3 * noise
    return l2_normalize(X)


def compute_ground_truth(queries: np.ndarray, base: np.ndarray,
                         k: int = 10, cache_dir: Path | None = None,
                         dataset_key: str = "") -> np.ndarray:
    cache_dir = cache_dir or (DATA_ROOT / "gt_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _sha(queries) + "_" + _sha(base) + f"_k{k}"
    if dataset_key:
        cache_key = dataset_key + "_" + cache_key
    cache_file = cache_dir / f"{cache_key}.npy"
    if cache_file.exists():
        return np.load(str(cache_file))
    gt = _brute_force_gt(queries, base, k)
    np.save(str(cache_file), gt)
    return gt


def _brute_force_gt(queries: np.ndarray, base: np.ndarray, k: int) -> np.ndarray:
    nq = len(queries)
    gt = np.empty((nq, k), dtype=np.int32)
    for i, q in enumerate(queries):
        sims = base @ q.astype(np.float32)
        top = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        gt[i] = top[:k].astype(np.int32)
    return gt


def recall_at_k(results: list[list[int]], ground_truth: np.ndarray, k: int) -> float:
    n = len(results)
    if n == 0:
        return 0.0
    total = 0.0
    for i, res in enumerate(results):
        gt_set = set(ground_truth[i, :k].tolist())
        total += sum(1 for r in res[:k] if r in gt_set) / len(gt_set) if gt_set else 0.0
    return total / n


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def dataset_checksum(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

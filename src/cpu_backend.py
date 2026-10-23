"""NumPy reference decoder for CPU performance measurement.

The implementation is unfused and operates in float32 precision.
Execution time includes Python dispatch, memory allocations, and materialized
attention matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from model import ModelConfig, RuntimeConfig


@dataclass
class KVCache:
    k: np.ndarray  # [layers, batch, kv_heads, max_seq, head_dim]
    v: np.ndarray
    length: int = 0

    @property
    def nbytes(self) -> int:
        return int(self.k.nbytes + self.v.nbytes)

    @property
    def capacity(self) -> int:
        return int(self.k.shape[3])


class NumpyDecoder:
    """Transformer decoder with RMSNorm, RoPE, GQA, and SwiGLU."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int = 0,
        max_seq_len: int | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.config = config
        self.dtype = np.dtype(dtype)
        self.head_dim = config.head_dim
        self.kv_dim = config.kv_dim
        self.n_rep = config.n_heads // config.n_kv_heads
        self.max_seq_len = max_seq_len or 2048
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        self._scale = 1.0 / math.sqrt(self.head_dim)
        rng = np.random.default_rng(seed)
        self.embed = (rng.standard_normal((config.vocab_size, config.hidden_size)) * 0.02).astype(self.dtype)
        self.final_norm = np.ones(config.hidden_size, dtype=self.dtype)
        self.layers: list[dict[str, np.ndarray]] = []
        for _ in range(config.n_layers):
            self.layers.append(self._make_layer(rng))
        if config.tie_embeddings:
            self.lm_head = self.embed
        else:
            self.lm_head = (rng.standard_normal((config.vocab_size, config.hidden_size)) * 0.02).astype(self.dtype)
        self.cos, self.sin = self._rope_tables(self.max_seq_len)

    def _make_layer(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.config
        d = cfg.hidden_size
        inter = cfg.intermediate_size
        kv_dim = cfg.kv_dim
        # SwiGLU packs gate and up projections into one [d, 2*I] matrix.
        mlp_mult = 2 if cfg.gated_mlp else 1

        def linear(rows: int, cols: int) -> np.ndarray:
            return (rng.standard_normal((rows, cols)) / math.sqrt(rows)).astype(self.dtype)

        return {
            "ln1": np.ones(d, dtype=self.dtype),
            "wq": linear(d, d),
            "wk": linear(d, kv_dim),
            "wv": linear(d, kv_dim),
            "wo": linear(d, d),
            "ln2": np.ones(d, dtype=self.dtype),
            "w_mlp": linear(d, mlp_mult * inter),
            "w_down": linear(inter, d),
        }

    def _rope_tables(self, max_seq: int) -> tuple[np.ndarray, np.ndarray]:
        half = self.head_dim // 2
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
        positions = np.arange(max_seq, dtype=np.float32)[:, None]
        freqs = positions * inv_freq[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(self.dtype), np.sin(emb).astype(self.dtype)

    def _rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        variance = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(variance + 1e-6)) * weight

    def _silu(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def _rope(self, x: np.ndarray, start: int) -> np.ndarray:
        # x: [batch, heads, time, head_dim]
        t = x.shape[2]
        cos = self.cos[start : start + t][None, None, :, :]
        sin = self.sin[start : start + t][None, None, :, :]
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = np.concatenate((-x2, x1), axis=-1)
        return x * cos + rotated * sin

    def _attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool
    ) -> np.ndarray:
        # q: [B, H, Tq, hd], k/v: [B, H, Tk, hd]
        scores = (q @ k.transpose(0, 1, 3, 2)) * self._scale
        if causal:
            tq, tk = scores.shape[-2], scores.shape[-1]
            mask = np.triu(np.ones((tq, tk), dtype=bool), k=1)
            scores = np.where(mask, -np.inf, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        return probs @ v

    def allocate_cache(self, batch_size: int) -> KVCache:
        cfg = self.config
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, self.max_seq_len, self.head_dim)
        return KVCache(
            k=np.zeros(shape, dtype=self.dtype),
            v=np.zeros(shape, dtype=self.dtype),
            length=0,
        )

    def prefill(self, tokens: np.ndarray) -> tuple[np.ndarray, KVCache]:
        """Process input tokens and initialize the KV cache."""
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, seq]")
        batch, seq = tokens.shape
        if seq > self.max_seq_len:
            raise ValueError("prefill longer than max_seq_len")
        cfg = self.config
        x = self.embed[tokens]  # [B, S, d]
        cache = self.allocate_cache(batch)
        for index, layer in enumerate(self.layers):
            h = self._rmsnorm(x, layer["ln1"])
            q = (h @ layer["wq"]).reshape(batch, seq, cfg.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = (h @ layer["wk"]).reshape(batch, seq, cfg.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = (h @ layer["wv"]).reshape(batch, seq, cfg.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            q = self._rope(q, 0)
            k = self._rope(k, 0)
            cache.k[index, :, :, :seq, :] = k
            cache.v[index, :, :, :seq, :] = v
            kk = np.repeat(k, self.n_rep, axis=1)
            vv = np.repeat(v, self.n_rep, axis=1)
            attn = self._attention(q, kk, vv, causal=True)
            attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.hidden_size)
            x = x + attn @ layer["wo"]

            h = self._rmsnorm(x, layer["ln2"])
            mlp = h @ layer["w_mlp"]
            if cfg.gated_mlp:
                gate, up = np.split(mlp, 2, axis=-1)
                mlp = self._silu(gate) * up
            else:
                mlp = self._silu(mlp)
            x = x + mlp @ layer["w_down"]
        cache.length = seq
        x = self._rmsnorm(x, self.final_norm)
        logits = x[:, -1, :] @ self.lm_head.T
        return logits, cache

    def decode_step(self, tokens: np.ndarray, cache: KVCache) -> np.ndarray:
        """Execute one decode step and update the KV cache."""
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        batch = tokens.shape[0]
        if cache.length >= cache.capacity:
            raise ValueError("KV cache is full")
        pos = cache.length
        cfg = self.config
        x = self.embed[tokens][:, None, :]  # [B, 1, d]
        for index, layer in enumerate(self.layers):
            h = self._rmsnorm(x, layer["ln1"])
            q = (h @ layer["wq"]).reshape(batch, 1, cfg.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = (h @ layer["wk"]).reshape(batch, 1, cfg.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = (h @ layer["wv"]).reshape(batch, 1, cfg.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            q = self._rope(q, pos)
            k = self._rope(k, pos)
            cache.k[index, :, :, pos, :] = k[:, :, 0, :]
            cache.v[index, :, :, pos, :] = v[:, :, 0, :]
            kk = np.repeat(cache.k[index, :, :, : pos + 1, :], self.n_rep, axis=1)
            vv = np.repeat(cache.v[index, :, :, : pos + 1, :], self.n_rep, axis=1)
            attn = self._attention(q, kk, vv, causal=False)
            attn = attn.transpose(0, 2, 1, 3).reshape(batch, 1, cfg.hidden_size)
            x = x + attn @ layer["wo"]

            h = self._rmsnorm(x, layer["ln2"])
            mlp = h @ layer["w_mlp"]
            if cfg.gated_mlp:
                gate, up = np.split(mlp, 2, axis=-1)
                mlp = self._silu(gate) * up
            else:
                mlp = self._silu(mlp)
            x = x + mlp @ layer["w_down"]
        cache.length += 1
        x = self._rmsnorm(x, self.final_norm)
        return x[:, 0, :] @ self.lm_head.T

    def weight_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        arrays: list[np.ndarray] = [self.embed, self.final_norm, self.lm_head]
        for layer in self.layers:
            arrays.extend(layer.values())
        for arr in arrays:
            if id(arr) in seen:
                continue
            seen.add(id(arr))
            total += int(arr.nbytes)
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "cpu-numpy-blas-float32",
            "config": self.config.name,
            "n_layers": self.config.n_layers,
            "hidden_size": self.config.hidden_size,
            "n_heads": self.config.n_heads,
            "n_kv_heads": self.config.n_kv_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "weight_bytes": self.weight_bytes(),
        }


def decoder_for_runtime(
    config: ModelConfig,
    runtime: RuntimeConfig,
    *,
    seed: int = 0,
    slack: int = 2,
) -> NumpyDecoder:
    return NumpyDecoder(
        config,
        seed=seed,
        max_seq_len=runtime.max_context + slack,
    )

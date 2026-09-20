"""Carga de audio: le, converte para mono, reamostra e normaliza."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AudioConfig, DEFAULT


@dataclass
class Audio:
    samples: np.ndarray
    """float32 mono em -1..1."""

    sample_rate: int
    path: Path | None = None
    duration: float = 0.0

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def slice(self, start: float, end: float) -> "Audio":
        a = max(0, int(start * self.sample_rate))
        b = min(self.n_samples, int(end * self.sample_rate))
        return Audio(self.samples[a:b], self.sample_rate, self.path,
                     max(0.0, (b - a) / self.sample_rate))


def load(path: str | Path, cfg: AudioConfig = DEFAULT.audio) -> Audio:
    """Carrega um arquivo de audio para analise.

    Reamostramos para cfg.sample_rate (22050 por padrao) porque nada do que
    fazemos - tempo, onset, f0 ate ~1.3 kHz - precisa de mais que isso, e o
    custo cai pela metade.
    """
    import librosa

    path = Path(path)
    y, sr = librosa.load(str(path), sr=cfg.sample_rate, mono=cfg.mono)
    y = np.asarray(y, dtype=np.float32)
    if cfg.normalize:
        y = normalize(y)
    return Audio(y, int(sr), path, len(y) / sr if sr else 0.0)


def from_samples(samples: np.ndarray, sample_rate: int,
                 cfg: AudioConfig = DEFAULT.audio) -> Audio:
    """Embrulha um array ja em memoria (usado pelos stems do Demucs)."""
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0 if y.shape[0] < y.shape[1] else 1)
    if cfg.normalize:
        y = normalize(y)
    return Audio(y, sample_rate, None, len(y) / sample_rate if sample_rate else 0.0)


def normalize(y: np.ndarray) -> np.ndarray:
    """Normaliza o pico para 1.0.

    Sem isso, os limiares de onset significariam coisas diferentes em uma
    musica masterizada alto e em uma gravacao de 1972.
    """
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / peak).astype(np.float32) if peak > 1e-9 else y


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-1 do arquivo. Chave do cache de stems.

    Hash do conteudo, nao do nome: renomear a musica nao deve custar 6 minutos
    de Demucs de novo.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

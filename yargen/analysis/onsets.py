"""Deteccao de onsets."""

from __future__ import annotations

import numpy as np

from ..audio.loader import Audio
from ..chart.ir import AnalyzedNote
from ..config import DEFAULT, OnsetConfig


def detect(audio: Audio, cfg: OnsetConfig = DEFAULT.onset) -> list[AnalyzedNote]:
    """Onsets de um stem, ja com forca normalizada em 0..1.

    A forca sai da envelope de onset no frame do pico, dividida pelo percentil
    `strength_percentile` da envelope. Percentil em vez de maximo porque um
    unico prato de bateria no refrao nao pode esmagar a escala do resto.
    """
    import librosa

    y, sr, hop = audio.samples, audio.sample_rate, cfg.hop_length
    if len(y) < hop * 4:
        return []

    # O detector precisa de `pre_avg` frames de contexto antes de aceitar um
    # pico, entao um ataque exatamente em t=0 e invisivel para ele. Isso custa
    # a primeira nota de qualquer musica que comece no tempo 1 sem intro.
    # Prefixamos silencio, detectamos, e descontamos o deslocamento.
    pad = int(round(cfg.pre_avg_ms / 1000.0 * sr)) + hop * 2
    y = np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    pad_sec = pad / sr

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, backtrack=cfg.backtrack,
        delta=cfg.delta, units="frames",
        pre_max=_frames(cfg.pre_max_ms, sr, hop), post_max=_frames(cfg.post_max_ms, sr, hop),
        pre_avg=_frames(cfg.pre_avg_ms, sr, hop), post_avg=_frames(cfg.post_avg_ms, sr, hop),
        wait=_frames(cfg.wait_ms, sr, hop))
    if len(frames) == 0:
        return []

    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    scale = float(np.percentile(env, cfg.strength_percentile)) or 1.0

    notes: list[AnalyzedNote] = []
    for frame, t in zip(frames, times):
        t = float(t) - pad_sec
        if t < -0.05:
            continue
        t = max(0.0, t)
        # Com backtrack o frame recua para o vale, onde a envelope e baixa por
        # definicao. A forca tem que vir do pico, nao do vale.
        peak = _local_peak(env, int(frame), _frames(cfg.post_max_ms, sr, hop))
        strength = float(np.clip(env[peak] / scale, 0.0, 1.0))
        if strength < cfg.min_strength:
            continue
        notes.append(AnalyzedNote(time=t, strength=strength))

    for i, note in enumerate(notes):
        nxt = notes[i + 1].time if i + 1 < len(notes) else audio.duration
        note.duration = max(0.0, nxt - note.time)
    return notes


def _frames(ms: float, sr: int, hop: int) -> int:
    return max(1, int(round(ms / 1000.0 * sr / hop)))


def _local_peak(env: np.ndarray, frame: int, radius: int) -> int:
    lo = max(0, frame)
    hi = min(len(env), frame + radius * 2 + 1)
    if hi <= lo:
        return min(max(frame, 0), len(env) - 1)
    return lo + int(np.argmax(env[lo:hi]))

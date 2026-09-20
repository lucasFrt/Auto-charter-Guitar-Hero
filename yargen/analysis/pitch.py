"""Contorno de f0 e atribuicao de altura a cada onset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio.loader import Audio
from ..chart.ir import AnalyzedNote
from ..config import DEFAULT, PitchConfig


@dataclass
class PitchContour:
    times: np.ndarray
    f0: np.ndarray
    """Hz; NaN onde nao ha altura confiavel."""

    voiced: np.ndarray

    def median_in(self, start: float, end: float) -> float | None:
        """Mediana do f0 no intervalo, ignorando frames sem altura.

        Mediana e nao media porque o pyin erra por oitava de vez em quando, e
        um unico salto de oitava arrasta a media mas nao a mediana.
        """
        lo, hi = np.searchsorted(self.times, (start, end))
        hi = max(hi, lo + 1)
        window = self.f0[lo:hi]
        window = window[np.isfinite(window)]
        return float(np.median(window)) if window.size else None

    def coverage(self, start: float, end: float) -> float:
        """Fracao de frames com altura no intervalo."""
        lo, hi = np.searchsorted(self.times, (start, end))
        hi = max(hi, lo + 1)
        window = self.f0[lo:hi]
        return float(np.mean(np.isfinite(window))) if window.size else 0.0


def contour(audio: Audio, cfg: PitchConfig = DEFAULT.pitch) -> PitchContour:
    import librosa

    y, sr = audio.samples, audio.sample_rate
    if len(y) < cfg.frame_length * 2:
        empty = np.zeros(0)
        return PitchContour(empty, empty, empty.astype(bool))

    if cfg.method == "pyin":
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=cfg.fmin_hz, fmax=cfg.fmax_hz, sr=sr,
            frame_length=cfg.frame_length, hop_length=cfg.hop_length)
        f0 = np.where((voiced_prob >= cfg.voiced_threshold) & np.isfinite(f0), f0, np.nan)
        voiced = np.isfinite(f0)
    else:
        f0 = librosa.yin(y, fmin=cfg.fmin_hz, fmax=cfg.fmax_hz, sr=sr,
                         frame_length=cfg.frame_length, hop_length=cfg.hop_length)
        f0 = np.where((f0 > cfg.fmin_hz) & (f0 < cfg.fmax_hz), f0, np.nan)
        voiced = np.isfinite(f0)

    if cfg.median_filter_frames > 1:
        f0 = _median_filter_semitones(f0, cfg.median_filter_frames)

    times = librosa.times_like(f0, sr=sr, hop_length=cfg.hop_length)
    return PitchContour(np.asarray(times), np.asarray(f0, dtype=float), voiced)


def _median_filter_semitones(f0: np.ndarray, k: int) -> np.ndarray:
    """Filtro de mediana em escala logaritmica.

    Em semitons e nao em Hz: um erro de meio tom em 80 Hz e 2.4 Hz e em 800 Hz
    e 24 Hz, entao filtrar em Hz trataria o agudo como se fosse muito mais
    instavel do que e.
    """
    out = f0.copy()
    semis = np.log2(np.where(np.isfinite(f0) & (f0 > 0), f0, np.nan))
    half = k // 2
    for i in range(len(f0)):
        lo, hi = max(0, i - half), min(len(f0), i + half + 1)
        window = semis[lo:hi]
        window = window[np.isfinite(window)]
        out[i] = 2.0 ** float(np.median(window)) if window.size else np.nan
    return out


def annotate(notes: list[AnalyzedNote], pitch: PitchContour,
             cfg: PitchConfig = DEFAULT.pitch, *,
             attack_skip: float = 0.02, max_span: float = 0.25) -> list[AnalyzedNote]:
    """Atribui pitch_hz a cada onset, medindo logo depois do ataque.

    `attack_skip` pula os primeiros milissegundos: o transiente de uma palheta
    e ruido de banda larga e o estimador de f0 se perde nele. `max_span` evita
    que uma nota longa engula a altura da frase inteira.
    """
    for note in notes:
        start = note.time + attack_skip
        end = start + min(max_span, note.duration if note.duration > 0 else max_span)
        note.pitch_hz = pitch.median_in(start, end)
    return notes

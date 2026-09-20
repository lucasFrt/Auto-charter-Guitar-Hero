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


# ------------------------------------------------------ deteccao por banda

PERCUSSIVE_BANDS: tuple[tuple[float, float], ...] = (
    (20.0, 120.0),      # bumbo
    (120.0, 400.0),     # corpo do bumbo, tom grave
    (400.0, 1600.0),    # caixa, palma, tom agudo
    (1600.0, 5000.0),   # chimbal fechado
    (5000.0, 11000.0),  # chimbal aberto, pratos
)
"""Cinco bandas, uma por traste. Nao e coincidencia: um kit de bateria se
organiza em faixas de frequencia que correspondem bem as cinco casas."""


def detect_percussive(audio: Audio, cfg: OnsetConfig = DEFAULT.onset,
                      bands: tuple[tuple[float, float], ...] = PERCUSSIVE_BANDS
                      ) -> list[AnalyzedNote]:
    """Onsets detectados INDEPENDENTEMENTE em cada banda de frequencia.

    Por que isto existe: numa bateria, tocar duas pecas ao mesmo tempo e a
    regra, nao a excecao - o chimbal marca todas as colcheias e o bumbo e a
    caixa caem por cima dele. Um detector de banda larga funde os dois num
    evento so, e o brilho medido nesse evento e uma mistura que nao
    corresponde a peca nenhuma: o bumbo desaparecia e o chart virava uma
    parede de chimbal.

    Detectando por banda, bumbo e chimbal simultaneos sao dois eventos com
    identidade inequivoca, e viram naturalmente um acorde - que e exatamente o
    que se quer tocar.

    O brilho atribuido e o centro geometrico da banda, nao o medido: assim o
    bumbo cai sempre no mesmo traste do primeiro ao ultimo compasso, que e o
    que torna a batida reconhecivel.
    """
    import librosa

    y, sr, hop = audio.samples, audio.sample_rate, cfg.hop_length
    if len(y) < hop * 4:
        return []

    pad = int(round(cfg.pre_avg_ms / 1000.0 * sr)) + hop * 2
    y = np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    pad_sec = pad / sr

    notes: list[AnalyzedNote] = []
    nyquist = sr / 2.0
    for lo_hz, hi_hz in bands:
        if lo_hz >= nyquist:
            continue
        hi_hz = min(hi_hz, nyquist * 0.99)
        env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop,
            S=librosa.power_to_db(librosa.feature.melspectrogram(
                y=y, sr=sr, hop_length=hop, fmin=lo_hz, fmax=hi_hz, n_mels=24)))
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=hop, backtrack=cfg.backtrack,
            delta=cfg.delta, units="frames",
            pre_max=_frames(cfg.pre_max_ms, sr, hop),
            post_max=_frames(cfg.post_max_ms, sr, hop),
            pre_avg=_frames(cfg.pre_avg_ms, sr, hop),
            post_avg=_frames(cfg.post_avg_ms, sr, hop),
            wait=_frames(cfg.wait_ms, sr, hop))
        if len(frames) == 0:
            continue

        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
        scale = float(np.percentile(env, cfg.strength_percentile)) or 1.0
        centre = float(np.sqrt(lo_hz * hi_hz))

        for frame, t in zip(frames, times):
            t = float(t) - pad_sec
            if t < -0.05:
                continue
            peak = _local_peak(env, int(frame), _frames(cfg.post_max_ms, sr, hop))
            strength = float(np.clip(env[peak] / scale, 0.0, 1.0))
            if strength < cfg.min_strength:
                continue
            notes.append(AnalyzedNote(time=max(0.0, t), strength=strength,
                                      centroid_hz=centre))

    notes = _collapse_adjacent_bands(notes, bands, cfg.wait_ms / 1000.0,
                                     cfg.collapse_band_radius)
    notes.sort(key=lambda n: (n.time, n.centroid_hz or 0.0))
    for i, note in enumerate(notes):
        nxt = notes[i + 1].time if i + 1 < len(notes) else audio.duration
        note.duration = max(0.0, nxt - note.time)
    return notes


def _collapse_adjacent_bands(notes: list[AnalyzedNote],
                             bands: tuple[tuple[float, float], ...],
                             window: float, radius: int = 2
                             ) -> list[AnalyzedNote]:
    """Funde deteccoes que sao a mesma pancada vista em duas bandas vizinhas.

    Uma peca de bateria nao mora numa banda so: um bumbo tem energia tambem
    em 200-400 Hz, um chimbal aparece nas duas bandas agudas. Sem tratar isso,
    cada pancada vira duas ou tres notas e o chart infla.

    A regra usa a DISTANCIA entre bandas para separar os dois casos:
      - bandas VIZINHAS no mesmo instante = uma peca so espalhada; fica a
        deteccao mais forte;
      - bandas DISTANTES no mesmo instante = pecas diferentes tocando juntas
        (bumbo + chimbal e o padrao do genero); ficam as duas, e o mapper as
        transforma no acorde que e exatamente o que se quer sentir.
    """
    if not notes:
        return notes
    centres = [float(np.sqrt(lo * hi)) for lo, hi in bands]

    def band_index(note: AnalyzedNote) -> int:
        target = note.centroid_hz or 0.0
        return min(range(len(centres)), key=lambda i: abs(centres[i] - target))

    ordered = sorted(notes, key=lambda n: n.time)
    groups: list[list[AnalyzedNote]] = [[ordered[0]]]
    for note in ordered[1:]:
        if note.time - groups[-1][0].time <= window:
            groups[-1].append(note)
        else:
            groups.append([note])

    out: list[AnalyzedNote] = []
    for group in groups:
        kept: list[AnalyzedNote] = []
        for note in sorted(group, key=lambda n: -n.strength):
            idx = band_index(note)
            if any(abs(idx - band_index(other)) <= radius for other in kept):
                continue
            kept.append(note)
        out.extend(kept)
    return out

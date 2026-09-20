"""Beat tracking e construcao da grade."""

from __future__ import annotations

import numpy as np

from ..audio.loader import Audio
from ..chart.ir import TempoMap
from ..config import DEFAULT, TempoConfig


def detect(audio: Audio, cfg: TempoConfig = DEFAULT.tempo, *,
           bpm_override: float | None = None) -> TempoMap:
    """Devolve a grade de beats da musica.

    Com `bpm_override`, gera uma grade rigida no BPM dado mas ainda usa o
    primeiro beat detectado como ancora - um BPM certo com a grade deslocada
    meio tempo e tao inutil quanto um BPM errado.
    """
    import librosa

    y, sr = audio.samples, audio.sample_rate

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length)
    tempo_est, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=cfg.hop_length,
        start_bpm=cfg.start_bpm, tightness=cfg.tightness, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=cfg.hop_length)
    beats = np.asarray(beats, dtype=float)

    if bpm_override is not None and bpm_override > 0:
        anchor = float(beats[0]) if len(beats) else 0.0
        return TempoMap.constant(bpm_override, audio.duration, cfg.resolution,
                                 cfg.beats_per_bar, start=anchor)

    if len(beats) < 2:
        # Beat tracking falhou (musica curta, ambiente, percussao ausente).
        # Cair para 120 BPM e melhor do que abortar: o usuario tem --bpm.
        return TempoMap.constant(cfg.start_bpm, max(audio.duration, 4.0),
                                 cfg.resolution, cfg.beats_per_bar)

    beats = _extend(beats, audio.duration, cfg.trim_leading_silence)
    downbeat = estimate_downbeat(onset_env, beat_frames, cfg.beats_per_bar)
    return TempoMap(list(beats), cfg.resolution, cfg.beats_per_bar, downbeat)


def _extend(beats: np.ndarray, duration: float, trim_leading: bool) -> np.ndarray:
    """Estende a grade ate o inicio e o fim do audio.

    O librosa so devolve beats onde houve energia. Se a musica tem 8 segundos
    de intro suave, as notas desse trecho cairiam fora da grade e teriam que
    ser extrapoladas nota a nota. Mais simples preencher a grade uma vez.
    """
    beats = list(map(float, beats))
    first_step = beats[1] - beats[0]
    if not trim_leading:
        while beats[0] - first_step > 0:
            beats.insert(0, beats[0] - first_step)
    last_step = beats[-1] - beats[-2]
    while beats[-1] + last_step <= duration + last_step:
        beats.append(beats[-1] + last_step)
    return np.asarray(beats, dtype=float)


def estimate_downbeat(onset_env: np.ndarray, beat_frames: np.ndarray,
                      beats_per_bar: int) -> int:
    """Qual dos primeiros `beats_per_bar` beats e o tempo forte.

    Heuristica barata: o tempo forte e, em media, o que tem mais energia de
    ataque. Somamos a forca de onset de cada classe de beat (i % 4) e pegamos
    a maior. Erra em musica sincopada, mas so afeta os marcadores de compasso
    e o bonus metrico da reducao de dificuldade, nunca a posicao das notas.
    """
    if len(beat_frames) < beats_per_bar * 2:
        return 0
    scores = np.zeros(beats_per_bar)
    for i, frame in enumerate(beat_frames):
        if 0 <= frame < len(onset_env):
            scores[i % beats_per_bar] += float(onset_env[frame])
    return int(np.argmax(scores))

"""Timbre: brilho por onset, para conteudo que nao tem altura.

Bateria nao tem f0. Perguntar "qual a altura deste bumbo" nao tem resposta, e
o pyin devolve ruido ou nada. Mas bateria tem uma escala perceptual obvia e
estavel: BRILHO. Bumbo e escuro, tom e medio-escuro, caixa e media, chimbal e
brilhante, prato e muito brilhante.

O centroide espectral mede exatamente isso, e mapeia direto em grave->agudo no
braco - a mesma intuicao que faz o mapeamento por altura funcionar para
guitarra. O mapper nao precisa saber a diferenca: ele recebe um escalar por
nota, em escala logaritmica de semitons, e distribui em faixas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio.loader import Audio
from ..chart.ir import AnalyzedNote
from ..config import DEFAULT, PitchConfig


@dataclass
class TimbreContour:
    times: np.ndarray
    centroid_hz: np.ndarray

    def median_in(self, start: float, end: float) -> float | None:
        lo, hi = np.searchsorted(self.times, (start, end))
        hi = max(hi, lo + 1)
        window = self.centroid_hz[lo:hi]
        window = window[np.isfinite(window) & (window > 0)]
        return float(np.median(window)) if window.size else None


def contour(audio: Audio, cfg: PitchConfig = DEFAULT.pitch) -> TimbreContour:
    import librosa

    y, sr, hop = audio.samples, audio.sample_rate, cfg.hop_length
    if len(y) < hop * 4:
        empty = np.zeros(0)
        return TimbreContour(empty, empty)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    times = librosa.times_like(centroid, sr=sr, hop_length=hop)
    return TimbreContour(np.asarray(times), np.asarray(centroid, dtype=float))


def annotate(notes: list[AnalyzedNote], timbre: TimbreContour, *,
             window: float = 0.06) -> list[AnalyzedNote]:
    """Mede o brilho logo no ataque de cada onset.

    Janela curta e DEPOIS do ataque, nao em volta dele: o que identifica a
    peca da bateria e o transiente. Meio segundo depois, o que sobra e o
    decaimento e a reverberacao da sala, que sao parecidos entre bumbo e
    caixa e confundiriam a classificacao.
    """
    for note in notes:
        note.centroid_hz = timbre.median_in(note.time, note.time + window)
    return notes

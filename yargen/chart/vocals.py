"""Vocals: alinhamento de letra + altura.

Duas fontes que precisam casar:
  - faster-whisper da os tempos por palavra (opcional; se nao estiver
    instalado, saem notas sem letra, que o YARG ainda aceita);
  - pyin sobre o stem vocal da o contorno de f0.

A juncao e por intervalo: para cada palavra, a altura e a mediana do f0
dentro do span da palavra. Onde nao ha f0 confiavel, a nota sai marcada como
nao-tonal ('#'), que faz o jogo cobrar so o ritmo. E melhor do que chutar uma
altura: altura errada em vocals e imediatamente frustrante.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import DEFAULT, Config, VocalsConfig
from .ir import (AnalyzedWord, Phrase, TempoMap, VocalChart, VocalNote,
                 hz_to_midi)


class TranscriptionUnavailable(RuntimeError):
    """Nao deu para transcrever. Carrega o motivo para o pipeline poder avisar."""


def transcribe(audio_path: str, cfg: VocalsConfig = DEFAULT.vocals) -> list[AnalyzedWord]:
    """Letra por palavra via faster-whisper.

    Levanta TranscriptionUnavailable em vez de propagar o erro original, e o
    pipeline cai para a segmentacao do contorno de f0. Nao e so o pacote estar
    ausente: na primeira execucao o faster-whisper BAIXA o modelo, e isso
    falha atras de proxy restritivo, sem rede, ou com o disco cheio. Um chart
    sem letra e muito melhor do que uma rodada de varios minutos perdida no
    ultimo passo.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionUnavailable(
            "faster-whisper nao esta instalado (pip install yargen[vocals])") from exc

    try:
        model = WhisperModel(cfg.whisper_model, device="cpu",
                             compute_type=cfg.whisper_compute_type)
        segments, _info = model.transcribe(audio_path, word_timestamps=True,
                                           language=cfg.whisper_language)
        words: list[AnalyzedWord] = []
        for segment in segments:
            for word in (segment.words or []):
                text = word.word.strip()
                if not text:
                    continue
                words.append(AnalyzedWord(time=float(word.start),
                                          duration=max(cfg.min_note_sec,
                                                       float(word.end) - float(word.start)),
                                          text=text))
        return words
    except Exception as exc:
        raise TranscriptionUnavailable(f"{type(exc).__name__}: {exc}") from exc


def attach_pitch(words: list[AnalyzedWord], contour, cfg: VocalsConfig) -> list[AnalyzedWord]:
    """Preenche a altura de cada palavra a partir do contorno de f0."""
    for word in words:
        end = word.time + word.duration
        coverage = contour.coverage(word.time, end)
        hz = contour.median_in(word.time, end)
        if hz is not None and coverage >= cfg.voiced_coverage_min:
            word.pitch_hz, word.pitched = hz, True
        else:
            word.pitch_hz, word.pitched = None, False
    return words


def words_from_contour(contour, cfg: VocalsConfig,
                       min_gap: float = 0.12,
                       split_semitones: float = 1.0,
                       confirm_frames: int = 2) -> list[AnalyzedWord]:
    """Fallback sem whisper: segmenta o contorno de f0 em notas.

    Quebra onde a voz para e onde a altura se afasta da nota que esta sendo
    sustentada. A comparacao e contra a MEDIANA do segmento corrente, nao
    contra o frame anterior: o contorno passa por um filtro de mediana, que
    espalha a transicao por varios frames, e um salto de tres semitons vira
    meio semitom por frame - invisivel para um teste frame a frame. Comparar
    com a mediana enxerga a mudanca sustentada, que e o que define uma nota
    nova.

    `confirm_frames` exige que o desvio persista, para vibrato e portamento
    nao picotarem uma nota longa em varias.

    Isto e o caminho degradado. Com faster-whisper instalado, as notas saem
    dos tempos por palavra, que sao bem melhores e ainda trazem a letra.
    """
    if contour.times.size == 0:
        return []
    import numpy as np

    words: list[AnalyzedWord] = []
    hop = float(contour.times[1] - contour.times[0]) if contour.times.size > 1 else 0.02

    def flush(a: int, b: int) -> None:
        if b <= a:
            return
        span = contour.f0[a:b]
        span = span[np.isfinite(span)]
        if not span.size:
            return
        t0 = float(contour.times[a])
        t1 = float(contour.times[min(b, len(contour.times) - 1)])
        if t1 - t0 < cfg.min_note_sec:
            return
        words.append(AnalyzedWord(t0, min(cfg.max_note_sec, t1 - t0), "",
                                  float(np.median(span)), True))

    start: int | None = None
    semis: list[float] = []
    pending = 0

    for i, value in enumerate(contour.f0):
        if not np.isfinite(value):
            if start is not None:
                flush(start, i)
            start, semis, pending = None, [], 0
            continue
        semi = 12.0 * float(np.log2(value / 440.0))
        if start is None:
            start, semis, pending = i, [semi], 0
            continue
        reference = float(np.median(semis))
        if abs(semi - reference) > split_semitones:
            pending += 1
            if pending >= confirm_frames:
                flush(start, i - pending + 1)
                start, semis, pending = i - pending + 1, [semi], 0
        else:
            pending = 0
            semis.append(semi)

    if start is not None:
        flush(start, len(contour.f0))
    return words


def build(words: Sequence[AnalyzedWord], tempo: TempoMap,
          cfg: Config = DEFAULT) -> VocalChart:
    """Monta o PART VOCALS a partir das palavras ja com altura."""
    v = cfg.vocals
    chart = VocalChart()
    if not words:
        return chart

    ordered = sorted(words, key=lambda w: w.time)
    for i, word in enumerate(ordered):
        duration = max(v.min_note_sec, min(v.max_note_sec, word.duration))
        # Nao deixa uma nota invadir a proxima: em vocals, notas sobrepostas
        # confundem o jogo e o display de letra.
        if i + 1 < len(ordered):
            duration = min(duration, max(v.min_note_sec, ordered[i + 1].time - word.time))

        midi_f = hz_to_midi(word.pitch_hz) if word.pitched else None
        if midi_f is None:
            midi, pitched = v.unpitched_fallback_midi, False
        else:
            midi = int(round(_fold_into_range(midi_f, v.pitch_min_midi, v.pitch_max_midi)))
            pitched = True

        start = tempo.time_to_tick(word.time)
        end = tempo.time_to_tick(word.time + duration)
        chart.notes.append(VocalNote(tick=max(0, int(round(start))),
                                     duration_ticks=max(1, int(round(end - start))),
                                     midi=midi, text=word.text, pitched=pitched))

    chart.phrases = build_phrases(chart.notes, tempo, v)
    return chart


def _fold_into_range(midi: float, lo: int, hi: int) -> float:
    """Traz a altura para 36..84 por oitavas, em vez de cortar no limite.

    Cortar transformaria um erro de oitava do pyin - que e comum - em uma nota
    colada no teto da faixa. Dobrar por oitava mantem a classe de altura, que
    e o que o jogador realmente canta.
    """
    while midi < lo:
        midi += 12
    while midi > hi:
        midi -= 12
    return max(lo, min(hi, midi))


def build_phrases(notes: Sequence[VocalNote], tempo: TempoMap,
                  cfg: VocalsConfig) -> list[Phrase]:
    """Marcadores de frase (nota 105).

    Sem frase, o YARG nao pontua vocals: as notas existem mas nao pertencem a
    nada. Quebramos em silencios maiores que `phrase_gap_sec` e a forca depois
    de `phrase_max_sec`, porque uma frase muito longa vira uma barra de
    progresso inutil na tela.
    """
    if not notes:
        return []
    phrases: list[Phrase] = []
    group: list[VocalNote] = [notes[0]]

    def close(items: list[VocalNote]) -> None:
        start = tempo.tick_to_time(items[0].tick) - cfg.phrase_pad_sec
        end = tempo.tick_to_time(items[-1].tick + items[-1].duration_ticks) + cfg.phrase_pad_sec
        phrases.append(Phrase(max(0, int(round(tempo.time_to_tick(start)))),
                              int(round(tempo.time_to_tick(end)))))

    for note in notes[1:]:
        prev = group[-1]
        gap = tempo.tick_to_time(note.tick) - \
            tempo.tick_to_time(prev.tick + prev.duration_ticks)
        span = tempo.tick_to_time(note.tick) - tempo.tick_to_time(group[0].tick)
        if gap > cfg.phrase_gap_sec or span > cfg.phrase_max_sec:
            close(group)
            group = [note]
        else:
            group.append(note)
    close(group)
    return phrases

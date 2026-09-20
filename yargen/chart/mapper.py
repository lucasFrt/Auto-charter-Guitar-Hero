"""O mapper: (tempo, tom, forca) -> trastes.

Este e o modulo que decide se o chart e gostoso ou intocavel. Tudo aqui e
funcao pura sobre a IR de analise: nao importa librosa, nao le arquivo, nao
tem estado global. E de proposito - e o que permite testar o mapper com JSON
fixo e iterar nos parametros sem re-rodar 6 minutos de Demucs.

Ordem dos estagios, e por que ela e essa:

O mapper NAO sabe de onde vem o "tom" de uma nota. Ele recebe um escalar por
nota, em escala logaritmica de semitons, e distribui em cinco faixas. Para
conteudo melodico o escalar e a altura (f0); para conteudo percussivo e o
brilho (centroide espectral), porque bateria nao tem altura mas tem uma escala
grave->agudo igualmente obvia. Essa indiferenca e o que permite chartear trap
- onde a trilha de 5 trastes e a BATIDA - com o mesmo codigo que charteia um
riff de rock.

Ordem dos estagios, e por que ela e essa:

  1. quantizar        - alinhar a grade antes de qualquer decisao de traste
  2. agrupar acordes  - onsets simultaneos viram um objeto so
  3. cortar densidade - ANTES de atribuir trastes, senao o limite de salto e
                        calculado sobre notas que vao ser jogadas fora
  4. atribuir trastes - faixas de percentil em janela movel + ancora
  5. limitar saltos   - playability ganha de fidelidade
  6. sustains
  7. HOPO
  8. star power
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import density
from ..config import DEFAULT, Config, MapperConfig
from .ir import (AnalysisIR, AnalyzedNote, ChartIR, ChartNote, InstrumentChart,
                 Phrase, TempoMap, hz_to_midi)

N_FRETS = 5


@dataclass
class _Event:
    """Nota em construcao: ja quantizada, ainda sem traste definitivo."""

    tick: int
    time: float
    strength: float
    tones: list[float]
    """Tom dos onsets que compoem este evento, em escala de semitons. Altura
    para conteudo melodico, brilho para percussivo - o mapper nao distingue."""

    frets: list[int]

    @property
    def tone(self) -> float | None:
        return float(np.median(self.tones)) if self.tones else None


# ============================================================ 1. quantizacao

def tone_from_pitch(note: AnalyzedNote) -> float | None:
    """Altura em semitons MIDI. Conteudo melodico."""
    return hz_to_midi(note.pitch_hz)


def tone_from_centroid(note: AnalyzedNote) -> float | None:
    """Brilho em semitons. Conteudo percussivo.

    Mesma escala logaritmica da altura de proposito: assim as tolerancias em
    "semitons" (ancora, limiar de acorde) significam a mesma coisa nos dois
    modos e nao precisam de um segundo conjunto de parametros.
    """
    return hz_to_midi(note.centroid_hz)


def quantize(notes: Sequence[AnalyzedNote], tempo: TempoMap,
             cfg: MapperConfig, tone_of=tone_from_pitch) -> list[_Event]:
    """Alinha os onsets a grade, mas nao a qualquer custo.

    Se a posicao quantizada cair a mais de `quantize_max_shift_ms` do onset
    real, mantemos a nota onde ela estava. Isso e uma valvula de seguranca
    contra o pior caso do M1: grade errada (rubato, gravacao ao vivo) puxando
    todas as notas para lugares que nao correspondem ao que se ouve. Melhor
    uma nota fora da grade do que uma nota no tempo errado.
    """
    events: list[_Event] = []
    for note in notes:
        snapped = tempo.quantize(note.time, cfg.grid_division)
        shift_ms = abs(tempo.tick_to_time(snapped) - note.time) * 1000.0
        tick = snapped if shift_ms <= cfg.quantize_max_shift_ms else \
            int(round(tempo.time_to_tick(note.time)))
        tone = tone_of(note)
        events.append(_Event(max(0, tick), note.time, note.strength,
                             [tone] if tone is not None else [], []))
    events.sort(key=lambda e: (e.tick, e.time))
    return events


# ============================================================== 2. acordes

def group_chords(events: list[_Event], tempo: TempoMap,
                 cfg: MapperConfig) -> list[_Event]:
    """Funde onsets simultaneos em um evento so.

    Onsets fracos nao entram em acorde (`chord_min_strength`): sem essa
    guarda, todo vazamento de bateria ou de outro instrumento no stem vira um
    acorde, e o chart fica com acordes onde a musica tem nota unica.
    """
    if not events:
        return []
    window_ticks = cfg.chord_window_ms / 1000.0 * tempo.resolution * \
        (tempo.average_bpm / 60.0 if tempo.average_bpm > 0 else 2.0)
    window_ticks = max(1.0, window_ticks)

    out: list[_Event] = []
    current = events[0]
    for event in events[1:]:
        close = (event.tick - current.tick) <= window_ticks
        strong_enough = min(event.strength, current.strength) >= cfg.chord_min_strength
        if close and strong_enough and cfg.chord_max_frets > 1:
            current.tones.extend(event.tones)
            current.strength = max(current.strength, event.strength)
        elif close:
            # Perto demais para serem duas notas, fraco demais para ser acorde:
            # fica a mais forte das duas.
            if event.strength > current.strength:
                current.strength = event.strength
                current.tones = event.tones or current.tones
        else:
            out.append(current)
            current = event
    out.append(current)
    return out


# ============================================================ 3. densidade

def cap_density(events: list[_Event], tempo: TempoMap, max_nps: float,
                window_sec: float, *, beat_bonus: float = 0.0,
                bar_bonus: float = 0.0) -> list[_Event]:
    """Corta as notas mais fracas ate respeitar o teto de notas/segundo.

    O score e a forca do onset mais um bonus por posicao metrica forte: entre
    duas notas de forca parecida, sobrevive a que cai no tempo do compasso.
    """
    if max_nps <= 0 or not events:
        return list(events)
    items = []
    for event in events:
        score = event.strength
        if bar_bonus and tempo.is_downbeat(event.tick):
            score += bar_bonus
        elif beat_bonus and tempo.is_on_beat(event.tick):
            score += beat_bonus
        items.append((tempo.tick_to_time(event.tick), score))
    return [events[i] for i in density.select(items, max_nps, window_sec)]


# ========================================================= 4. pitch -> traste

class _Anchors:
    """Memoria de "este tom ja virou este traste".

    Preserva riffs: se a musica volta ao mesmo si bemol tres compassos depois,
    o jogador reencontra a mesma casa e o padrao continua reconhecivel no
    highway. Sem isso, a janela movel de percentis reclassifica a mesma nota
    conforme o contexto muda e o riff se desfaz.
    """

    def __init__(self, window_sec: float, tolerance: float) -> None:
        self.window = window_sec
        self.tolerance = tolerance
        self._items: list[tuple[float, float, int]] = []  # (time, pitch, fret)

    def lookup(self, time: float, tone: float) -> int | None:
        self._items = [it for it in self._items if time - it[0] <= self.window]
        best, best_dist = None, self.tolerance
        for t, p, fret in reversed(self._items):
            dist = abs(p - tone)
            if dist <= best_dist:
                best, best_dist = fret, dist
        return best

    def record(self, time: float, tone: float, fret: int) -> None:
        self._items.append((time, tone, fret))


def _band_edges(tones: Sequence[float], percentiles: Sequence[float]) -> np.ndarray:
    """Cortes de percentil calculados sobre as alturas DISTINTAS da janela.

    Sobre as alturas repetidas os percentis empatam e as faixas colapsam: um
    riff que martela a tonica 8 vezes e toca 3 outras notas coloca varios
    cortes em cima da tonica, e os trastes do meio deixam de existir. O riff
    sai do mapper achatado em duas ou tres cores.

    Arredondar para semitom antes de tirar os distintos tambem evita que
    vibrato ou um bend leve virem "alturas diferentes" e reintroduzam o
    mesmo empate por outro caminho.
    """
    values = np.unique(np.round(np.asarray(tones, dtype=float)))
    if values.size < 2:
        values = np.unique(np.asarray(tones, dtype=float))
    return np.percentile(values, list(percentiles))


ROLLING = "rolling"
GLOBAL = "global"


def assign_frets(events: list[_Event], tempo: TempoMap, cfg: MapperConfig,
                 *, band_scope: str = ROLLING) -> list[_Event]:
    """Atribui trastes a partir do tom.

    `band_scope` decide sobre o que os percentis sao calculados:

      ROLLING - janela movel de `band_window_sec`. Para conteudo melodico: o
        registro muda entre estrofe e refrao, e um solo uma oitava acima da
        estrofe deve usar os cinco trastes tambem, nao virar "tudo laranja".

      GLOBAL - a musica inteira de uma vez. Para conteudo percussivo: um kit
        de bateria tem quatro ou cinco sons e eles nao mudam. O bumbo precisa
        ser verde no compasso 1 e no compasso 80. Uma janela movel
        reclassificaria o bumbo conforme o arranjo fica mais cheio ou mais
        vazio, e a batida deixaria de ser reconhecivel - que e justamente o
        que se quer tocar.
    """
    toned = [(i, e) for i, e in enumerate(events) if e.tone is not None]
    if not toned:
        return assign_frets_random(events, cfg)

    # Achatado: uma entrada por TOM, nao por evento. Usar a mediana do evento
    # aqui envenenava as faixas - num acorde de bumbo+chimbal a mediana cai no
    # meio, onde nao ha peca nenhuma, e os cortes de percentil saiam deslocados
    # a ponto de a caixa ser classificada junto com o bumbo.
    flat_times = np.array([e.time for _, e in toned for _ in e.tones])
    flat_tones = np.array([t for _, e in toned for t in e.tones])

    anchors = _Anchors(cfg.anchor_window_sec, cfg.anchor_tolerance_semitones)
    half = cfg.band_window_sec / 2.0
    global_edges = _band_edges(flat_tones, cfg.band_percentiles) \
        if band_scope == GLOBAL else None

    last_fret = 2
    for k, (idx, event) in enumerate(toned):
        tone = event.tone
        assert tone is not None

        if cfg.anchor_enabled and len(event.tones) == 1:
            anchored = anchors.lookup(event.time, tone)
            if anchored is not None:
                event.frets = [anchored]
                last_fret = anchored
                continue

        if global_edges is not None:
            edges = global_edges
        else:
            lo = int(np.searchsorted(flat_times, event.time - half))
            hi = int(np.searchsorted(flat_times, event.time + half))
            if hi - lo < cfg.band_min_notes:
                # Janela pobre (inicio da musica, trecho esparso): expande por
                # indice ate ter estatistica suficiente em vez de calcular
                # percentis em cima de duas notas.
                need = cfg.band_min_notes
                centre = int(np.searchsorted(flat_times, event.time))
                lo = max(0, centre - need // 2)
                hi = min(len(flat_tones), lo + need)
                lo = max(0, hi - need)
            window = flat_tones[lo:hi]
            edges = _band_edges(window, cfg.band_percentiles) \
                if window.size >= 2 and float(np.ptp(window)) >= 1e-6 else None

        if edges is None:
            event.frets = [last_fret]
        else:
            # Um traste POR TOM. Sem isto, um evento com dois tons - que e o
            # que um acorde E - era reduzido a mediana dos dois e virava uma
            # nota unica em algum lugar no meio. Acorde nenhum saia do mapper,
            # em nenhuma estrategia; `shape_chords` recebia sempre uma lista
            # de um elemento e nao tinha o que moldar.
            frets = sorted({max(0, min(N_FRETS - 1,
                                       int(np.searchsorted(edges, t, side="right"))))
                            for t in event.tones})
            event.frets = frets or [last_fret]

        last_fret = event.frets[0]
        if cfg.anchor_enabled and len(event.tones) == 1:
            anchors.record(event.time, tone, event.frets[0])

    # Eventos sem tom herdam o traste do vizinho anterior: repetir uma nota
    # e sempre jogavel, e melhor do que inventar.
    prev = 2
    for event in events:
        if event.frets:
            prev = event.frets[0]
        else:
            event.frets = [prev]
    return events


def assign_frets_random(events: list[_Event], cfg: MapperConfig) -> list[_Event]:
    """Trastes pseudoaleatorios respeitando o limite de salto (modo M1).

    Continua util depois do M3: quando a deteccao de f0 nao achou nada (stem
    percussivo, distorcao pesada), isto produz um chart que pelo menos cai no
    tempo, que e melhor do que nao produzir nada.
    """
    rng = random.Random(cfg.seed)
    prev = rng.randrange(N_FRETS)
    for event in events:
        lo = max(0, prev - cfg.max_jump)
        hi = min(N_FRETS - 1, prev + cfg.max_jump)
        choices = [f for f in range(lo, hi + 1) if f != prev] or [prev]
        prev = rng.choice(choices)
        event.frets = [prev]
    return events


# ========================================================= 5. limite de salto

def limit_jumps(events: list[_Event], tempo: TempoMap,
                cfg: MapperConfig) -> list[_Event]:
    """Impede saltos grandes em sequencia rapida.

    Entre notas separadas por menos de `jump_window_ms`, a mao nao tem tempo
    de atravessar o braco: limitamos o salto a `max_jump` trastes, puxando a
    nota na direcao certa (preservando o contorno melodico) em vez de
    substitui-la por um valor arbitrario.
    """
    if not events:
        return events
    prev = events[0]
    for event in events[1:]:
        dt_ms = (tempo.tick_to_time(event.tick) - tempo.tick_to_time(prev.tick)) * 1000.0
        if dt_ms < cfg.jump_window_ms:
            anchor = max(prev.frets)
            clamped = []
            for fret in event.frets:
                delta = fret - anchor
                if abs(delta) > cfg.max_jump:
                    fret = anchor + cfg.max_jump * (1 if delta > 0 else -1)
                clamped.append(max(0, min(N_FRETS - 1, fret)))
            event.frets = sorted(set(clamped))
        prev = event
    return events


def shape_chords(events: list[_Event], cfg: MapperConfig) -> list[_Event]:
    """Reduz cada acorde a trastes tocaveis.

    No maximo `chord_max_frets` casas e, por padrao, adjacentes. Um acorde de
    verde + laranja e literalmente uma mao aberta no braco inteiro; ninguem
    quer isso em um chart gerado automaticamente.
    """
    for event in events:
        if len(event.tones) <= 1:
            event.frets = event.frets[:1] or [2]
            continue
        frets = sorted(set(event.frets))
        if len(frets) == 1:
            continue
        if len(frets) > cfg.chord_max_frets:
            # Mais tons do que casas permitidas: fica o mais grave e o mais
            # agudo, que sao os que definem a forma do acorde.
            frets = [frets[0], frets[-1]]
        frets = frets[:cfg.chord_max_frets]
        if cfg.chord_adjacent_only and len(frets) == 2 and frets[1] - frets[0] > 1:
            low = min(frets[0], N_FRETS - 2)
            frets = [low, low + 1]
        event.frets = sorted(set(frets))
    return events


# ============================================================== 6/7. sustains e hopo

def build_notes(events: list[_Event], tempo: TempoMap,
                cfg: MapperConfig) -> list[ChartNote]:
    """Converte eventos em ChartNote, calculando sustain e HOPO."""
    res = tempo.resolution
    min_sustain_beats = cfg.sustain_min_beats
    cutoff = res // 3  # abaixo disso o jogo descarta o sustain de qualquer jeito

    notes: list[ChartNote] = []
    for i, event in enumerate(events):
        # A ultima nota nao tem proxima, e nao sabemos se ela soa ate o fim da
        # musica ou para seco. Antes o valor de fallback era exatamente o
        # limiar de sustain, entao a ultima nota SEMPRE sustentava - inclusive
        # uma pancada de bumbo, que nunca sustenta. Sem evidencia, nao sustenta;
        # a cauda final e um ajuste de dez segundos no Moonscraper.
        if i + 1 >= len(events):
            notes.append(ChartNote(tick=event.tick, frets=list(event.frets),
                                   sustain_ticks=0, strength=event.strength))
            continue
        gap = events[i + 1].tick - event.tick
        sustain = 0
        if gap >= min_sustain_beats * res:
            sustain = int(min(gap * cfg.sustain_ratio, cfg.sustain_max_beats * res))
            if sustain < cutoff:
                sustain = 0
        notes.append(ChartNote(tick=event.tick, frets=list(event.frets),
                               sustain_ticks=sustain, strength=event.strength))

    if cfg.hopo_enabled:
        threshold = cfg.hopo_threshold_ticks if cfg.hopo_threshold_ticks is not None \
            else res // 3 + 1
        from .writer_mid import compute_natural_hopo
        for note, hopo in zip(notes, compute_natural_hopo(notes, threshold)):
            note.hopo = hopo
    return notes


# ============================================================== 8. star power

def build_star_power(notes: Sequence[ChartNote], tempo: TempoMap,
                     cfg: MapperConfig) -> list[Phrase]:
    """Escolhe frases de Star Power.

    Quebramos o chart em frases nos silencios maiores que `sp_phrase_gap_sec`,
    descartamos as curtas demais e pegamos frases espacadas de
    `sp_interval_sec`. Nao e musical - Star Power humano marca o refrao - mas
    distribui o recurso de forma jogavel ao longo da musica, que e o minimo
    para o chart nao ser frustrante.
    """
    if not cfg.star_power_enabled or not notes:
        return []

    groups: list[list[ChartNote]] = [[notes[0]]]
    for note in notes[1:]:
        gap = tempo.tick_to_time(note.tick) - tempo.tick_to_time(groups[-1][-1].tick)
        (groups[-1] if gap <= cfg.sp_phrase_gap_sec else groups.append([]) or groups[-1]).append(note)

    phrases: list[Phrase] = []
    last_time = -1e9
    for group in groups:
        if len(group) < cfg.sp_min_notes:
            continue
        start_time = tempo.tick_to_time(group[0].tick)
        if start_time - last_time < cfg.sp_interval_sec:
            continue
        end = group[-1].tick + max(group[-1].sustain_ticks, tempo.resolution // 4)
        phrases.append(Phrase(max(0, group[0].tick - 1), end))
        last_time = start_time
        if len(phrases) >= cfg.sp_max_phrases:
            break
    return phrases


# ==================================================================== api

def map_track(notes: Sequence[AnalyzedNote], tempo: TempoMap,
              cfg: Config = DEFAULT, *, use_pitch: bool = True,
              strategy: str = "pitch") -> InstrumentChart:
    """Pipeline completo de um stem para um chart Expert.

    `strategy` escolhe de onde vem o tom e como as faixas sao calculadas:
      "pitch"      - altura, faixas em janela movel (melodico)
      "percussive" - brilho, faixas globais (bateria, percussao)
    """
    m = cfg.mapper
    percussive = strategy == "percussive"
    tone_of = tone_from_centroid if percussive else tone_from_pitch
    scope = GLOBAL if percussive else ROLLING

    events = quantize(notes, tempo, m, tone_of)
    events = group_chords(events, tempo, m)
    events = cap_density(events, tempo, m.max_notes_per_second, m.density_window_sec,
                         beat_bonus=cfg.difficulty.beat_bonus,
                         bar_bonus=cfg.difficulty.bar_bonus)
    if use_pitch and any(e.tone is not None for e in events):
        events = assign_frets(events, tempo, m, band_scope=scope)
    else:
        events = assign_frets_random(events, m)
    events = shape_chords(events, m)
    events = limit_jumps(events, tempo, m)
    chart_notes = build_notes(events, tempo, m)
    return InstrumentChart("PART GUITAR", notes={"expert": chart_notes},
                           star_power=build_star_power(chart_notes, tempo, m))

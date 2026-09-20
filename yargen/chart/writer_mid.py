"""Escrita do notes.mid.

Todos os numeros aqui saem da documentacao do TheNathannator
(GuitarGame_ChartFormats, docs/Chart-File-Formats/mid-format/), que e a
referencia que o proprio YARG usa. Onde a doc e explicita sobre um formato ser
"o recomendado", seguimos o recomendado e nao o mais novo.

Pontos que custaram leitura e vale registrar:

  - As dificuldades sao oitavas: Easy 60, Medium 72, Hard 84, Expert 96, e
    dentro de cada oitava o layout e sempre o mesmo (base+0..4 = cores,
    base+5 = force HOPO, base+6 = force strum, base-1 = nota aberta).
  - HOPO e AUTOMATICO no jogo: uma nota vira HOPO se estiver a no maximo
    (resolution / 3) + 1 ticks da anterior, nao for a mesma casa e nao for
    acorde. Por isso nao gravamos marcador de forcing em tudo: calculamos o
    estado natural e so gravamos 101/102 quando queremos o contrario dele.
  - Sustains menores que resolution/3 sao descartados pelo jogo. Notas "secas"
    saem com um comprimento minimo so para existir um note_off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import mido

from ..config import Config, DEFAULT
from .ir import (DIFFICULTIES, OPEN, ChartIR, ChartNote, InstrumentChart,
                 Phrase, TempoMap, VocalChart)

# ------------------------------------------------------- constantes do formato

DIFF_BASE = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}
"""Nota MIDI da primeira casa (verde) de cada dificuldade."""

OFFSET_FORCE_HOPO = 5
OFFSET_FORCE_STRUM = 6
OFFSET_OPEN = -1

NOTE_STAR_POWER = 116
NOTE_SOLO = 103
NOTE_TAP = 104

# PART VOCALS
VOX_PHRASE_P1 = 105
VOX_PHRASE_P2 = 106
VOX_STAR_POWER = 116
VOX_MIN, VOX_MAX = 36, 84

# BEAT
BEAT_MEASURE = 12
BEAT_STRONG = 13
BEAT_WEAK = 14

# SysEx Phase Shift: 'P' 'S' 0x00 <type> <diff> <modifier> <value>
PS_HEADER = (0x50, 0x53, 0x00)
PS_TYPE_PHRASE = 0x00
PS_DIFF = {"easy": 0x00, "medium": 0x01, "hard": 0x02, "expert": 0x03}
PS_MOD_OPEN = 0x01
PS_MOD_TAP = 0x04
PS_VALUE_END, PS_VALUE_START = 0x00, 0x01

DEFAULT_VELOCITY = 100

# Ordem de desempate quando varios eventos caem no mesmo tick. note_off antes
# de note_on evita que o fim de uma nota apague o inicio da seguinte na mesma
# casa; meta/sysex no meio para que uma frase abra depois de fechar a anterior.
_RANK = {"note_off": 0, "meta": 1, "sysex": 2, "note_on": 3}


class _EventList:
    """Acumula eventos em tempo absoluto e emite uma MidiTrack ordenada."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._events: list[tuple[int, int, int, mido.BaseMessage]] = []
        self._seq = 0

    def add(self, tick: int, msg: mido.BaseMessage, kind: str) -> None:
        if tick < 0:
            tick = 0
        self._events.append((tick, _RANK[kind], self._seq, msg))
        self._seq += 1

    def note(self, tick: int, pitch: int, length: int, velocity: int = DEFAULT_VELOCITY,
             channel: int = 0) -> None:
        length = max(1, int(length))
        self.add(tick, mido.Message("note_on", note=pitch, velocity=velocity,
                                    channel=channel, time=0), "note_on")
        self.add(tick + length, mido.Message("note_off", note=pitch, velocity=0,
                                             channel=channel, time=0), "note_off")

    def text(self, tick: int, value: str, kind: str = "text") -> None:
        self.add(tick, mido.MetaMessage(kind, text=value, time=0), "meta")

    def sysex(self, tick: int, data: Sequence[int]) -> None:
        self.add(tick, mido.Message("sysex", data=list(data), time=0), "sysex")

    def build(self) -> mido.MidiTrack:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=self.name, time=0))
        last = 0
        for tick, _rank, _seq, msg in sorted(self._events, key=lambda e: e[:3]):
            msg = msg.copy(time=tick - last)
            track.append(msg)
            last = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        return track

    def __len__(self) -> int:
        return len(self._events)


# ------------------------------------------------------------------ mecanicas

def natural_hopo_threshold(resolution: int) -> int:
    """(resolution / 3) + 1, arredondado para baixo.

    E o limiar moderno documentado. A 480 ticks da 161, que equivale a uma
    colcheia de tercina mais um tick de folga.
    """
    return resolution // 3 + 1


def compute_natural_hopo(notes: Sequence[ChartNote], threshold: int) -> list[bool]:
    """Estado que o jogo daria a cada nota sem nenhum marcador de forcing.

    Regra: vira HOPO se estiver perto o bastante da anterior, nao for acorde e
    nao repetir a casa da anterior. A primeira nota nunca e HOPO.
    """
    out: list[bool] = []
    prev: ChartNote | None = None
    for note in notes:
        if prev is None or len(note.frets) > 1:
            out.append(False)
        elif note.tick - prev.tick > threshold:
            out.append(False)
        elif len(prev.frets) == 1 and prev.frets[0] == note.frets[0]:
            out.append(False)
        else:
            out.append(True)
        prev = note
    return out


# -------------------------------------------------------------------- trilhas

def _write_instrument(chart: InstrumentChart, cfg: Config, resolution: int) -> mido.MidiTrack:
    ev = _EventList(chart.track_name)
    threshold = (cfg.mapper.hopo_threshold_ticks
                 if cfg.mapper.hopo_threshold_ticks is not None
                 else natural_hopo_threshold(resolution))
    dry = max(1, resolution // max(1, cfg.output.zero_sustain_divisor))
    use_note_opens = cfg.output.open_note_style == "note"
    needs_enhanced_opens = False

    for diff in DIFFICULTIES:
        notes = sorted(chart.notes.get(diff, []), key=lambda n: n.tick)
        if not notes:
            continue
        base = DIFF_BASE[diff]
        natural = compute_natural_hopo(notes, threshold)

        for note, is_natural_hopo in zip(notes, natural):
            length = note.sustain_ticks if note.sustain_ticks > 0 else dry

            for fret in note.frets:
                if fret == OPEN:
                    if use_note_opens:
                        needs_enhanced_opens = True
                        ev.note(note.tick, base + OFFSET_OPEN, length)
                    else:
                        # Frase SysEx cobrindo a nota: tudo dentro dela vira
                        # aberta. E o metodo que a doc recomenda escrever.
                        _ps_phrase(ev, note.tick, note.tick + length, diff, PS_MOD_OPEN)
                        ev.note(note.tick, base, length)
                else:
                    ev.note(note.tick, base + fret, length)

            # Forcing so quando o desejado diverge do natural.
            if cfg.mapper.hopo_enabled and note.hopo != is_natural_hopo:
                marker = base + (OFFSET_FORCE_HOPO if note.hopo else OFFSET_FORCE_STRUM)
                ev.note(note.tick, marker, length)

            if note.tap:
                _ps_phrase(ev, note.tick, note.tick + length, diff, PS_MOD_TAP)

    # Frases valem para todas as dificuldades: uma marcacao so.
    for phrase in chart.star_power:
        ev.note(phrase.start_tick, NOTE_STAR_POWER,
                max(1, phrase.end_tick - phrase.start_tick))
    for phrase in chart.solos:
        ev.note(phrase.start_tick, NOTE_SOLO,
                max(1, phrase.end_tick - phrase.start_tick))

    if needs_enhanced_opens:
        # Precisa estar no comeco da trilha para habilitar notas abertas
        # baseadas em nota; sem isso o jogo as ignora (59 e animacao em RB).
        ev.text(0, "[ENHANCED_OPENS]")

    return ev.build()


def _ps_phrase(ev: _EventList, start: int, end: int, diff: str, modifier: int) -> None:
    d = PS_DIFF[diff]
    ev.sysex(start, [*PS_HEADER, PS_TYPE_PHRASE, d, modifier, PS_VALUE_START])
    ev.sysex(max(start + 1, end), [*PS_HEADER, PS_TYPE_PHRASE, d, modifier, PS_VALUE_END])


def _write_vocals(vox: VocalChart, cfg: Config) -> mido.MidiTrack:
    ev = _EventList(vox.track_name)

    for note in sorted(vox.notes, key=lambda n: n.tick):
        pitch = max(VOX_MIN, min(VOX_MAX, note.midi))
        length = max(1, note.duration_ticks)
        text = note.text
        if text and not note.pitched and not text.endswith(("#", "^")):
            # '#' marca a silaba como nao-tonal: o jogo cobra o ritmo, nao a
            # altura. E o que queremos quando o pyin nao achou f0 confiavel.
            text += "#"
        if text:
            ev.add(note.tick, mido.MetaMessage("lyrics", text=text, time=0), "meta")
        ev.note(note.tick, pitch, length)

    for phrase in vox.phrases:
        ev.note(phrase.start_tick, VOX_PHRASE_P1,
                max(1, phrase.end_tick - phrase.start_tick))
    for phrase in vox.star_power:
        ev.note(phrase.start_tick, VOX_STAR_POWER,
                max(1, phrase.end_tick - phrase.start_tick))

    return ev.build()


def _write_beat(tempo: TempoMap) -> mido.MidiTrack:
    ev = _EventList("BEAT")
    res, per_bar = tempo.resolution, tempo.beats_per_bar
    length = max(1, res // 8)
    for i in range(len(tempo.beats)):
        tick = i * res
        strong = ((i - tempo.downbeat_index) % per_bar) == 0
        ev.note(tick, BEAT_MEASURE if strong else BEAT_STRONG, length)
    return ev.build()


def _write_events(ir: ChartIR, tempo: TempoMap) -> mido.MidiTrack:
    ev = _EventList("EVENTS")
    for tick, name in sorted(ir.sections):
        ev.text(tick, f"[section {name}]")
    end_tick = int(round(tempo.time_to_tick(ir.meta.duration))) if ir.meta.duration else \
        (len(tempo.beats) - 1) * tempo.resolution
    ev.text(max(0, end_tick), "[end]")
    return ev.build()


def _write_tempo_track(ir: ChartIR, tempo: TempoMap) -> mido.MidiTrack:
    ev = _EventList(ir.meta.name or "yargen")
    for tick, usec in tempo.tempo_events(ir_min_drift(ir)):
        ev.add(tick, mido.MetaMessage("set_tempo", tempo=usec, time=0), "meta")
    ev.add(0, mido.MetaMessage("time_signature", numerator=tempo.beats_per_bar,
                               denominator=4, time=0), "meta")
    return ev.build()


def ir_min_drift(ir: ChartIR) -> float:
    return 0.5


# ---------------------------------------------------------------------- api

def write_midi(ir: ChartIR, path: str | Path, cfg: Config = DEFAULT) -> mido.MidiFile:
    """Escreve `ir` como notes.mid.

    Formato 1 (multi-trilha) e resolucao em ticks-por-seminima, que e o unico
    que os jogos aceitam: tipo 0/2 e SMPTE nao carregam.
    """
    if ir.tempo is None:
        raise ValueError("ChartIR sem TempoMap: nao da para escrever MIDI")
    tempo = ir.tempo
    mid = mido.MidiFile(type=1, ticks_per_beat=tempo.resolution)

    mid.tracks.append(_write_tempo_track(ir, tempo))

    if cfg.output.write_events_track:
        mid.tracks.append(_write_events(ir, tempo))
    if cfg.output.write_beat_track:
        mid.tracks.append(_write_beat(tempo))

    for chart in ir.instruments.values():
        mid.tracks.append(_write_instrument(chart, cfg, tempo.resolution))

    if ir.vocals is not None and ir.vocals.notes:
        mid.tracks.append(_write_vocals(ir.vocals, cfg))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return mid

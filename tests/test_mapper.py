"""Testes do mapper.

Todos deterministicos e sem audio: a entrada e IR construida na mao ou lida do
JSON fixo em tests/data. E exatamente o contrato que a IR existe para permitir.
"""

import pytest

from yargen.chart.ir import AnalyzedNote, TempoMap, FRET_NAMES
from yargen.chart.mapper import (assign_frets, assign_frets_random, cap_density,
                                 group_chords, limit_jumps, map_track, quantize,
                                 shape_chords)
from yargen.config import DEFAULT


def notes_at(times, pitch=220.0, strength=0.8):
    return [AnalyzedNote(time=t, strength=strength, pitch_hz=pitch) for t in times]


def frets_of(chart_notes):
    return [n.frets for n in chart_notes]


def letters(chart_notes):
    return "".join(FRET_NAMES[n.frets[0]] for n in chart_notes)


# ------------------------------------------------------------- quantizacao

def test_quantize_snaps_near_misses_to_the_grid():
    tm = TempoMap.constant(120.0, 10.0)
    events = quantize(notes_at([0.508, 1.012]), tm, DEFAULT.mapper)
    assert [e.tick for e in events] == [480, 960]


def test_quantize_keeps_notes_that_are_too_far_from_the_grid():
    """Valvula de seguranca contra grade errada.

    Se o beat tracking errou, puxar a nota para a grade coloca ela num lugar
    que nao corresponde ao que se ouve. Manter fora da grade e menos pior.
    """
    tm = TempoMap.constant(120.0, 10.0)
    cfg = DEFAULT.override_path("mapper.quantize_max_shift_ms", 20.0).mapper
    # 0.560s fica a 60ms do 1/16 mais proximo com grade de 1/16 a 120bpm? nao:
    # usamos uma grade grossa para forcar o desvio.
    cfg2 = DEFAULT.override_path("mapper.quantize_max_shift_ms", 20.0)
    cfg2 = cfg2.override_path("mapper.grid_division", 4).mapper
    events = quantize(notes_at([0.62]), tm, cfg2)
    assert events[0].tick != 480 and events[0].tick != 960


# ------------------------------------------------------------------ acordes

def test_simultaneous_onsets_become_one_chord():
    tm = TempoMap.constant(120.0, 10.0)
    raw = [AnalyzedNote(0.0, 0.9, 220.0), AnalyzedNote(0.01, 0.9, 277.2)]
    events = group_chords(quantize(raw, tm, DEFAULT.mapper), tm, DEFAULT.mapper)
    assert len(events) == 1 and len(events[0].tones) == 2


def test_weak_onsets_do_not_form_chords():
    """Vazamento de outro instrumento no stem nao pode virar acorde."""
    tm = TempoMap.constant(120.0, 10.0)
    raw = [AnalyzedNote(0.0, 0.9, 220.0), AnalyzedNote(0.01, 0.10, 277.2)]
    events = group_chords(quantize(raw, tm, DEFAULT.mapper), tm, DEFAULT.mapper)
    assert len(events) == 1 and len(events[0].tones) == 1


def test_chords_are_limited_to_two_adjacent_frets():
    tm = TempoMap.constant(120.0, 10.0)
    raw = [AnalyzedNote(0.0, 0.9, 110.0), AnalyzedNote(0.005, 0.9, 880.0)]
    chart = map_track(raw, tm, DEFAULT)
    frets = chart.notes["expert"][0].frets
    assert len(frets) <= 2
    if len(frets) == 2:
        assert frets[1] - frets[0] == 1, "acorde de casas nao adjacentes e intocavel"


# ------------------------------------------------------------ limite de salto

def test_fast_notes_never_jump_more_than_max_jump():
    tm = TempoMap.constant(200.0, 20.0)
    # notas a 75ms: dentro da jump_window de 150ms
    raw = [AnalyzedNote(i * 0.075, 0.8, 110.0 * (8 ** (i % 2)))
           for i in range(24)]
    chart = map_track(raw, tm, DEFAULT)
    picked = [n.frets[0] for n in chart.notes["expert"]]
    deltas = [abs(b - a) for a, b in zip(picked, picked[1:])]
    assert max(deltas) <= DEFAULT.mapper.max_jump


def test_slow_notes_are_allowed_to_jump_freely():
    tm = TempoMap.constant(60.0, 40.0)
    raw = [AnalyzedNote(i * 1.0, 0.8, 110.0 * (8 ** (i % 2))) for i in range(12)]
    chart = map_track(raw, tm, DEFAULT)
    picked = [n.frets[0] for n in chart.notes["expert"]]
    assert max(abs(b - a) for a, b in zip(picked, picked[1:])) > DEFAULT.mapper.max_jump


# ----------------------------------------------------------------- ancoras

def test_repeated_riff_maps_to_the_same_frets_every_time(riff_ir):
    """Criterio do M3: um riff reconhecivel continua reconhecivel no highway."""
    tm = riff_ir.tempo
    chart = map_track(riff_ir.tracks["guitar"].notes, tm, DEFAULT)
    picked = letters(chart.notes["expert"])
    assert len(picked) == 32
    first = picked[:8]
    assert picked == first * 4, f"riff nao se repetiu: {picked}"


def test_riff_uses_more_than_two_frets(riff_ir):
    """Regressao: percentis sobre alturas repetidas empatavam e as faixas
    colapsavam, achatando o riff em duas ou tres cores."""
    chart = map_track(riff_ir.tracks["guitar"].notes, riff_ir.tempo, DEFAULT)
    assert len({n.frets[0] for n in chart.notes["expert"]}) >= 4


def test_anchor_can_be_disabled():
    cfg = DEFAULT.override_path("mapper.anchor_enabled", False)
    tm = TempoMap.constant(120.0, 30.0)
    raw = [AnalyzedNote(i * 0.25, 0.8, 220.0 * 2 ** ((i % 5) / 12)) for i in range(40)]
    assert map_track(raw, tm, cfg).notes["expert"]


# ------------------------------------------------------- janela de percentil

def test_long_run_sweeps_the_neck_not_just_the_middle():
    """A janela movel se auto-centra: numa corrida ascendente longa, cada nota
    fica no meio da propria janela. Com a janela padrao a corrida tem que
    varrer os cinco trastes, nao virar um bloco de amarelo."""
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.25, 0.8, 440 * 2 ** ((i - 12) / 12)) for i in range(24)]
    picked = [n.frets[0] for n in map_track(raw, tm, DEFAULT).notes["expert"]]
    assert len(set(picked)) == 5
    assert picked == sorted(picked), "corrida ascendente deve subir no braco"
    # nenhum traste pode concentrar mais da metade da corrida
    assert max(picked.count(f) for f in set(picked)) <= len(picked) // 2


def test_window_adapts_between_sections_in_different_registers():
    tm = TempoMap.constant(120.0, 80.0)
    raw = []
    t = 0.0
    for hz in [165, 196, 220, 196] * 12:       # secao grave, ~12s
        raw.append(AnalyzedNote(t, 0.8, hz)); t += 0.25
    t += 2.0
    for hz in [440, 494, 523, 494] * 12:       # secao aguda, ~12s
        raw.append(AnalyzedNote(t, 0.8, hz)); t += 0.25
    picked = [n.frets[0] for n in map_track(raw, tm, DEFAULT).notes["expert"]]
    low, high = picked[:48], picked[-48:]
    assert len(set(low)) >= 3, "secao grave devia usar varias cores, nao so o verde"
    assert len(set(high)) >= 3, "secao aguda devia usar varias cores, nao so o laranja"


# --------------------------------------------------------------- densidade

def test_density_cap_is_respected():
    tm = TempoMap.constant(240.0, 30.0)
    raw = [AnalyzedNote(i * 0.05, 0.3 + (i % 7) / 10, 220.0) for i in range(400)]
    cfg = DEFAULT.override_path("mapper.max_notes_per_second", 6.0)
    notes = map_track(raw, tm, cfg).notes["expert"]
    span = tm.tick_to_time(notes[-1].tick) - tm.tick_to_time(notes[0].tick)
    assert len(notes) / span <= 6.0 * 1.05


def test_density_cap_keeps_the_strongest_notes():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.05, 1.0 if i % 10 == 0 else 0.2, 220.0)
           for i in range(200)]
    cfg = DEFAULT.override_path("mapper.max_notes_per_second", 2.0)
    kept = map_track(raw, tm, cfg).notes["expert"]
    assert sum(1 for n in kept if n.strength > 0.9) >= 15


# ------------------------------------------------------------------ sustain

def test_long_gap_becomes_a_sustain():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(0.0, 0.9, 220.0), AnalyzedNote(2.0, 0.9, 220.0)]
    notes = map_track(raw, tm, DEFAULT).notes["expert"]
    assert notes[0].sustain_ticks > 0


def test_short_gap_has_no_sustain():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.25, 0.9, 220.0) for i in range(8)]
    notes = map_track(raw, tm, DEFAULT).notes["expert"]
    assert all(n.sustain_ticks == 0 for n in notes[:-1])


def test_sustains_never_overlap_the_next_note():
    tm = TempoMap.constant(120.0, 40.0)
    raw = [AnalyzedNote(t, 0.9, 220.0) for t in (0.0, 2.0, 2.5, 6.0, 10.0)]
    notes = map_track(raw, tm, DEFAULT).notes["expert"]
    for a, b in zip(notes, notes[1:]):
        assert a.tick + a.sustain_ticks <= b.tick


# -------------------------------------------------------------------- hopo

def test_close_notes_on_different_frets_are_hopo():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.125, 0.8, 220 * 2 ** (i / 12)) for i in range(8)]
    notes = map_track(raw, tm, DEFAULT).notes["expert"]
    assert any(n.hopo for n in notes[1:])


def test_first_note_is_never_hopo():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.125, 0.8, 220 * 2 ** (i / 12)) for i in range(8)]
    assert not map_track(raw, tm, DEFAULT).notes["expert"][0].hopo


# ------------------------------------------------------------ modo aleatorio

def test_random_mode_is_deterministic_and_respects_jumps():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.25, 0.8) for i in range(40)]
    a = [n.frets for n in map_track(raw, tm, DEFAULT, use_pitch=False).notes["expert"]]
    b = [n.frets for n in map_track(raw, tm, DEFAULT, use_pitch=False).notes["expert"]]
    assert a == b, "mesma semente tem que dar o mesmo chart"
    flat = [f[0] for f in a]
    assert max(abs(y - x) for x, y in zip(flat, flat[1:])) <= DEFAULT.mapper.max_jump


def test_notes_without_pitch_still_produce_a_chart():
    tm = TempoMap.constant(120.0, 20.0)
    raw = [AnalyzedNote(i * 0.25, 0.8, None) for i in range(20)]
    assert len(map_track(raw, tm, DEFAULT).notes["expert"]) == 20


def test_empty_input_produces_empty_chart():
    assert map_track([], TempoMap.constant(120.0, 10.0), DEFAULT).notes["expert"] == []

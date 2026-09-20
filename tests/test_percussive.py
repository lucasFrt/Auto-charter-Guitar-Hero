"""Mapeamento percussivo: brilho -> traste, com faixas globais."""

import collections

import pytest

from yargen.analysis.onsets import PERCUSSIVE_BANDS, _collapse_adjacent_bands
from yargen.chart.ir import AnalyzedNote, TempoMap
from yargen.chart.mapper import map_track, tone_from_centroid, tone_from_pitch
from yargen.config import DEFAULT

KICK, SNARE, HAT = 60.0, 800.0, 7000.0
TRAP = DEFAULT.override_path("mapper.anchor_enabled", False) \
              .override_path("mapper.chord_adjacent_only", False)


def beat(bars=8, bpm=120.0):
    """Batida sintetica: bumbo em 1 e 3, caixa em 2 e 4, chimbal em colcheias."""
    step = 60.0 / bpm
    notes = []
    for bar in range(bars):
        base = bar * 4 * step
        for b in (0, 2):
            notes.append(AnalyzedNote(base + b * step, 0.95, centroid_hz=KICK))
        for b in (1, 3):
            notes.append(AnalyzedNote(base + b * step, 0.85, centroid_hz=SNARE))
        for i in range(8):
            notes.append(AnalyzedNote(base + i * step * 0.5, 0.5, centroid_hz=HAT))
    return sorted(notes, key=lambda n: n.time)


def test_tone_comes_from_centroid_not_pitch():
    note = AnalyzedNote(0.0, 1.0, pitch_hz=None, centroid_hz=440.0)
    assert tone_from_pitch(note) is None
    assert tone_from_centroid(note) == pytest.approx(69.0, abs=0.01)


def test_drum_pieces_get_distinct_frets():
    tm = TempoMap.constant(120.0, 30.0)
    notes = map_track(beat(), tm, TRAP, strategy="percussive").notes["expert"]
    assert len({f for n in notes for f in n.frets}) >= 3


def test_the_kick_lands_on_the_same_fret_from_start_to_finish():
    """Faixas GLOBAIS existem para isto. Com janela movel, o bumbo seria
    reclassificado conforme o arranjo enche ou esvazia, e a batida deixaria de
    ser reconhecivel - que e justamente o que se quer tocar."""
    tm = TempoMap.constant(120.0, 40.0)
    raw = beat(bars=8)
    notes = map_track(raw, tm, TRAP, strategy="percussive").notes["expert"]

    kick_ticks = {tm.quantize(n.time, 16) for n in raw if n.centroid_hz == KICK}
    kick_frets = collections.Counter(
        tuple(n.frets) for n in notes if n.tick in kick_ticks)
    assert kick_frets, "nenhum bumbo sobreviveu"
    dominant = kick_frets.most_common(1)[0][1]
    assert dominant / sum(kick_frets.values()) >= 0.8


def test_the_kick_is_lower_on_the_neck_than_the_hat():
    """Grave -> agudo tem que virar verde -> laranja, senao a intuicao quebra."""
    tm = TempoMap.constant(120.0, 40.0)
    raw = [AnalyzedNote(i * 0.5, 0.9, centroid_hz=KICK) for i in range(16)]
    raw += [AnalyzedNote(i * 0.5 + 0.25, 0.9, centroid_hz=HAT) for i in range(16)]
    raw.sort(key=lambda n: n.time)
    notes = map_track(raw, tm, TRAP, strategy="percussive").notes["expert"]
    kicks = [n.frets[0] for n, r in zip(notes, raw) if r.centroid_hz == KICK]
    hats = [n.frets[0] for n, r in zip(notes, raw) if r.centroid_hz == HAT]
    assert max(kicks) < min(hats)


def test_percussive_notes_do_not_sustain():
    tm = TempoMap.constant(120.0, 40.0)
    cfg = TRAP.override_path("mapper.sustain_min_beats", 4.0)
    notes = map_track(beat(), tm, cfg, strategy="percussive").notes["expert"]
    assert all(n.sustain_ticks == 0 for n in notes)


# --------------------------------------------------- colapso entre bandas

def banded(time, band_index, strength=0.8):
    lo, hi = PERCUSSIVE_BANDS[band_index]
    return AnalyzedNote(time, strength, centroid_hz=(lo * hi) ** 0.5)


def test_same_hit_seen_in_neighbouring_bands_collapses():
    """Uma peca de bateria nao mora numa banda so: sem colapsar, cada pancada
    vira duas ou tres notas e o chart infla."""
    notes = [banded(1.0, 2, 0.9), banded(1.005, 3, 0.4)]
    kept = _collapse_adjacent_bands(notes, PERCUSSIVE_BANDS, 0.03, radius=2)
    assert len(kept) == 1 and kept[0].strength == 0.9


def test_distant_bands_survive_as_a_chord():
    """Bumbo + chimbal simultaneos e a assinatura do trap e tem que virar
    acorde, nao ser jogado fora junto com os duplicados."""
    notes = [banded(1.0, 0, 0.9), banded(1.005, 4, 0.7)]
    kept = _collapse_adjacent_bands(notes, PERCUSSIVE_BANDS, 0.03, radius=2)
    assert len(kept) == 2


def test_hits_far_apart_in_time_are_never_collapsed():
    notes = [banded(1.0, 2, 0.9), banded(2.0, 3, 0.4)]
    kept = _collapse_adjacent_bands(notes, PERCUSSIVE_BANDS, 0.03, radius=2)
    assert len(kept) == 2


def test_collapse_keeps_the_strongest_detection():
    notes = [banded(1.0, 2, 0.3), banded(1.002, 3, 0.95)]
    kept = _collapse_adjacent_bands(notes, PERCUSSIVE_BANDS, 0.03, radius=2)
    assert len(kept) == 1 and kept[0].strength == 0.95

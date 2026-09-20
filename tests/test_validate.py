"""Harness de precisao/recall."""

import pytest

from yargen.chart.ir import ChartIR, ChartNote, InstrumentChart, SongMeta, TempoMap
from yargen.chart.writer_mid import write_midi
from yargen.config import DEFAULT
from yargen.validate.compare import compare, compare_files, note_times

REF = [0.0, 0.5, 1.0, 1.5, 2.0]


def test_identical_scores_perfect():
    s = compare(REF, REF)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0


def test_small_offset_still_matches_within_tolerance():
    assert compare([t + 0.02 for t in REF], REF).f1 == 1.0


def test_offset_beyond_tolerance_matches_nothing():
    assert compare([t + 0.08 for t in REF], REF).matched == 0


def test_missing_notes_lower_recall_not_precision():
    s = compare(REF[::2], REF)
    assert s.precision == 1.0 and s.recall < 1.0


def test_spurious_notes_lower_precision_not_recall():
    s = compare(sorted(REF + [t + 0.25 for t in REF]), REF)
    assert s.recall == 1.0 and s.precision < 1.0


def test_matches_are_exclusive():
    """Sem exclusividade, um trecho denso 'acertaria' a mesma nota humana
    varias vezes e o recall mentiria para cima."""
    s = compare([1.0, 1.01, 1.02, 1.03], [1.0])
    assert s.matched == 1 and s.recall == 1.0 and s.precision == 0.25


def test_empty_inputs_do_not_crash():
    assert compare([], REF).f1 == 0.0
    assert compare(REF, []).f1 == 0.0


def test_round_trip_through_a_real_midi_file(tmp_path):
    """Le de volta um notes.mid que nos mesmos escrevemos e confere os
    instantes, inclusive a conversao tick->segundos."""
    tm = TempoMap.constant(120.0, 20.0)
    notes = [ChartNote(i * 480, [i % 5]) for i in range(8)]
    ir = ChartIR(meta=SongMeta(name="t", duration=20.0), tempo=tm,
                 instruments={"PART GUITAR": InstrumentChart(
                     "PART GUITAR", notes={"expert": notes})})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    times = note_times(tmp_path / "notes.mid", "expert")
    assert times == pytest.approx([i * 0.5 for i in range(8)], abs=1e-6)
    assert compare_files(tmp_path / "notes.mid", tmp_path / "notes.mid").f1 == 1.0


def test_reads_variable_tempo_files_correctly(tmp_path):
    beats, step = [0.0], 0.5
    for _ in range(20):
        beats.append(beats[-1] + step)
        step *= 0.95
    tm = TempoMap(beats)
    notes = [ChartNote(i * 480, [0]) for i in range(10)]
    ir = ChartIR(meta=SongMeta(name="t", duration=beats[-1]), tempo=tm,
                 instruments={"PART GUITAR": InstrumentChart(
                     "PART GUITAR", notes={"expert": notes})})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    times = note_times(tmp_path / "notes.mid", "expert")
    assert times == pytest.approx(beats[:10], abs=2e-3)

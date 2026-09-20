"""IR: serializacao fiel nos dois sentidos."""

import pytest

from yargen.chart.ir import (AnalysisIR, AnalyzedNote, AnalyzedTrack, AnalyzedWord,
                             ChartIR, ChartNote, InstrumentChart, OPEN, Phrase,
                             SongMeta, TempoMap, VocalChart, VocalNote,
                             hz_to_midi, midi_to_hz)


def test_analysis_ir_round_trips(tmp_path):
    ir = AnalysisIR(
        meta=SongMeta(name="n", artist="a", duration=12.5),
        tempo=TempoMap.constant(174.0, 12.5),
        tracks={"guitar": AnalyzedTrack("guitar", [
            AnalyzedNote(0.5, 0.8, 220.0, 0.25),
            AnalyzedNote(0.75, 0.4, None, 0.25)], source_stem="other"),
            "vocals": AnalyzedTrack("vocals", [], [
                AnalyzedWord(1.0, 0.3, "ola", 261.6, True)])})
    ir.save(tmp_path / "ir.json")
    back = AnalysisIR.load(tmp_path / "ir.json")
    assert back.to_dict() == ir.to_dict()
    assert back.tracks["guitar"].notes[1].pitch_hz is None
    assert back.tracks["vocals"].words[0].text == "ola"


def test_chart_ir_round_trips(tmp_path):
    ir = ChartIR(
        meta=SongMeta(name="n", duration=30.0),
        tempo=TempoMap.constant(120.0, 30.0),
        instruments={"PART GUITAR": InstrumentChart("PART GUITAR", notes={
            "expert": [ChartNote(0, [0, 1], 240, True), ChartNote(480, [OPEN])]},
            star_power=[Phrase(0, 480)])},
        vocals=VocalChart(notes=[VocalNote(0, 120, 60, "a", True)],
                          phrases=[Phrase(0, 240)]),
        sections=[(0, "intro"), (1920, "verso")])
    ir.save(tmp_path / "chart.json")
    back = ChartIR.load(tmp_path / "chart.json")
    assert back.to_dict() == ir.to_dict()
    assert back.instruments["PART GUITAR"].notes["expert"][0].hopo is True
    assert back.sections == [(0, "intro"), (1920, "verso")]


def test_note_count_summary():
    ir = ChartIR(instruments={"PART GUITAR": InstrumentChart(
        "PART GUITAR", notes={"expert": [ChartNote(0, [0])] * 3,
                              "easy": [ChartNote(0, [0])]})})
    assert ir.note_count() == {"PART GUITAR:expert": 3, "PART GUITAR:easy": 1}


@pytest.mark.parametrize("hz,midi", [(440.0, 69.0), (261.626, 60.0), (880.0, 81.0)])
def test_hz_midi_conversion(hz, midi):
    assert hz_to_midi(hz) == pytest.approx(midi, abs=0.01)
    assert midi_to_hz(midi) == pytest.approx(hz, rel=1e-4)


def test_hz_to_midi_handles_missing_and_zero():
    assert hz_to_midi(None) is None
    assert hz_to_midi(0.0) is None

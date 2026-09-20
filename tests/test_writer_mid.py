"""Writer MIDI: os numeros do formato, conferidos contra a doc canonica.

Referencia: TheNathannator/GuitarGame_ChartFormats,
docs/Chart-File-Formats/mid-format/.
"""

import mido
import pytest

from yargen.chart.ir import (ChartIR, ChartNote, InstrumentChart, OPEN, Phrase,
                             SongMeta, TempoMap, VocalChart, VocalNote)
from yargen.chart.writer_mid import (compute_natural_hopo, natural_hopo_threshold,
                                     write_midi)
from yargen.config import DEFAULT


def build_ir(notes_by_diff, **kw):
    tm = TempoMap.constant(120.0, 30.0)
    gtr = InstrumentChart("PART GUITAR", notes=notes_by_diff, **kw)
    return ChartIR(meta=SongMeta(name="t", duration=30.0), tempo=tm,
                   instruments={"PART GUITAR": gtr})


def read_track(path, name):
    mid = mido.MidiFile(str(path))
    for track in mid.tracks:
        if any(m.type == "track_name" and m.name == name for m in track):
            out, tick = [], 0
            for msg in track:
                tick += msg.time
                out.append((tick, msg))
            return out
    raise AssertionError(f"trilha {name} nao encontrada")


def note_ons(events, lo=0, hi=127):
    return [(t, m.note) for t, m in events
            if m.type == "note_on" and m.velocity > 0 and lo <= m.note <= hi]


# ------------------------------------------------------------ numeros base

@pytest.mark.parametrize("diff,base", [("expert", 96), ("hard", 84),
                                       ("medium", 72), ("easy", 60)])
def test_each_difficulty_uses_its_own_octave(tmp_path, diff, base):
    # Espacadas: cinco notas no mesmo tick em casas diferentes nao seriam uma
    # sequencia, seriam um acorde, e o writer (corretamente) gravaria
    # marcadores de forcing para elas.
    ir = build_ir({diff: [ChartNote(f * 480, [f]) for f in range(5)]})
    path = tmp_path / "notes.mid"
    write_midi(ir, path, DEFAULT)
    got = sorted(n for _, n in note_ons(read_track(path, "PART GUITAR")))
    assert got == list(range(base, base + 5))


def test_file_is_type_1_with_tick_resolution(tmp_path):
    """Tipo 0/2 e resolucao SMPTE simplesmente nao carregam nos jogos."""
    ir = build_ir({"expert": [ChartNote(0, [0])]})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    mid = mido.MidiFile(str(tmp_path / "notes.mid"))
    assert mid.type == 1 and mid.ticks_per_beat == 480


def test_star_power_and_solo_markers(tmp_path):
    ir = build_ir({"expert": [ChartNote(0, [0])]},
                  star_power=[Phrase(0, 960)], solos=[Phrase(960, 1920)])
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    got = dict((n, t) for t, n in note_ons(read_track(tmp_path / "notes.mid",
                                                      "PART GUITAR")))
    assert 116 in got and 103 in got


def test_chord_writes_simultaneous_notes(tmp_path):
    ir = build_ir({"expert": [ChartNote(480, [0, 1])]})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    got = note_ons(read_track(tmp_path / "notes.mid", "PART GUITAR"))
    assert sorted(got) == [(480, 96), (480, 97)]


def test_sustain_length_is_preserved(tmp_path):
    ir = build_ir({"expert": [ChartNote(0, [0], sustain_ticks=720)]})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    events = read_track(tmp_path / "notes.mid", "PART GUITAR")
    off = [t for t, m in events if m.type == "note_off" and m.note == 96]
    assert off == [720]


def test_dry_note_is_shorter_than_the_game_sustain_cutoff(tmp_path):
    """Nota sem sustain precisa de um note_off, mas o comprimento tem que
    ficar abaixo de resolution/3 para o jogo nao mostrar um rastro."""
    ir = build_ir({"expert": [ChartNote(0, [0])]})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    events = read_track(tmp_path / "notes.mid", "PART GUITAR")
    off = [t for t, m in events if m.type == "note_off" and m.note == 96][0]
    assert 0 < off < 480 // 3


# -------------------------------------------------------------------- hopo

def test_natural_hopo_threshold_matches_the_spec():
    """(resolution / 3) + 1, o limiar moderno documentado."""
    assert natural_hopo_threshold(480) == 161
    assert natural_hopo_threshold(192) == 65


def test_natural_hopo_rules():
    notes = [ChartNote(0, [0]), ChartNote(120, [1]), ChartNote(240, [1]),
             ChartNote(1000, [2]), ChartNote(1100, [0, 1])]
    assert compute_natural_hopo(notes, 161) == [False, True, False, False, False]


def test_no_forcing_marker_when_desired_matches_natural(tmp_path):
    """O jogo ja faz HOPO sozinho. Gravar 101/102 em tudo seria ruido e
    mudaria o hash do chart sem mudar como ele se joga."""
    notes = [ChartNote(0, [0]), ChartNote(120, [1], hopo=True)]
    write_midi(build_ir({"expert": notes}), tmp_path / "notes.mid", DEFAULT)
    got = [n for _, n in note_ons(read_track(tmp_path / "notes.mid", "PART GUITAR"))]
    assert 101 not in got and 102 not in got


def test_force_strum_marker_when_we_want_to_override_natural_hopo(tmp_path):
    notes = [ChartNote(0, [0]), ChartNote(120, [1], hopo=False)]
    write_midi(build_ir({"expert": notes}), tmp_path / "notes.mid", DEFAULT)
    got = [n for _, n in note_ons(read_track(tmp_path / "notes.mid", "PART GUITAR"))]
    assert 102 in got, "precisa forcar strum: o natural seria HOPO"


def test_force_hopo_marker_when_natural_would_be_strum(tmp_path):
    notes = [ChartNote(0, [0]), ChartNote(960, [1], hopo=True)]
    write_midi(build_ir({"expert": notes}), tmp_path / "notes.mid", DEFAULT)
    got = [n for _, n in note_ons(read_track(tmp_path / "notes.mid", "PART GUITAR"))]
    assert 101 in got


# ------------------------------------------------------------ notas abertas

def test_open_note_via_sysex_is_the_default(tmp_path):
    """A doc recomenda o SysEx do Phase Shift: o marcador por nota e mais
    novo e ainda nao e suportado em toda parte."""
    write_midi(build_ir({"expert": [ChartNote(0, [OPEN])]}),
               tmp_path / "notes.mid", DEFAULT)
    events = read_track(tmp_path / "notes.mid", "PART GUITAR")
    sysex = [m for _, m in events if m.type == "sysex"]
    assert len(sysex) == 2
    assert list(sysex[0].data)[:3] == [0x50, 0x53, 0x00]      # 'P' 'S' '\0'
    assert list(sysex[0].data)[3:] == [0x00, 0x03, 0x01, 0x01]  # frase, expert, open, inicio
    assert list(sysex[1].data)[3:] == [0x00, 0x03, 0x01, 0x00]  # ... fim


def test_open_note_via_note_requires_enhanced_opens(tmp_path):
    cfg = DEFAULT.override_path("output.open_note_style", "note")
    write_midi(build_ir({"expert": [ChartNote(0, [OPEN])]}),
               tmp_path / "notes.mid", cfg)
    events = read_track(tmp_path / "notes.mid", "PART GUITAR")
    assert 95 in [n for _, n in note_ons(events)]
    texts = [m.text for _, m in events if m.type == "text"]
    assert "[ENHANCED_OPENS]" in texts


# ------------------------------------------------------------------- vocals

def test_vocals_track_has_lyrics_pitch_and_phrases(tmp_path):
    tm = TempoMap.constant(120.0, 30.0)
    vox = VocalChart(notes=[VocalNote(0, 240, 60, "ho"), VocalNote(240, 240, 64, "la")],
                     phrases=[Phrase(0, 480)])
    ir = ChartIR(meta=SongMeta(name="t", duration=30.0), tempo=tm, vocals=vox)
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    events = read_track(tmp_path / "notes.mid", "PART VOCALS")
    assert [n for _, n in note_ons(events, 36, 84)] == [60, 64]
    assert 105 in [n for _, n in note_ons(events)]
    assert [m.text for _, m in events if m.type == "lyrics"] == ["ho", "la"]


def test_unpitched_syllable_is_marked_with_hash(tmp_path):
    tm = TempoMap.constant(120.0, 30.0)
    vox = VocalChart(notes=[VocalNote(0, 240, 60, "ye", pitched=False)])
    ir = ChartIR(meta=SongMeta(name="t", duration=30.0), tempo=tm, vocals=vox)
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    events = read_track(tmp_path / "notes.mid", "PART VOCALS")
    assert [m.text for _, m in events if m.type == "lyrics"] == ["ye#"]


def test_vocal_pitch_is_clamped_to_the_legal_range(tmp_path):
    tm = TempoMap.constant(120.0, 30.0)
    vox = VocalChart(notes=[VocalNote(0, 240, 12, "a"), VocalNote(240, 240, 120, "b")])
    ir = ChartIR(meta=SongMeta(name="t", duration=30.0), tempo=tm, vocals=vox)
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    got = [n for _, n in note_ons(read_track(tmp_path / "notes.mid", "PART VOCALS"))]
    assert all(36 <= n <= 84 for n in got)


# ------------------------------------------------------------ outras trilhas

def test_beat_track_marks_downbeats_differently(tmp_path):
    write_midi(build_ir({"expert": [ChartNote(0, [0])]}), tmp_path / "notes.mid", DEFAULT)
    got = [n for _, n in note_ons(read_track(tmp_path / "notes.mid", "BEAT"))]
    assert got[:8] == [12, 13, 13, 13, 12, 13, 13, 13]


def test_events_track_has_end_marker(tmp_path):
    write_midi(build_ir({"expert": [ChartNote(0, [0])]}), tmp_path / "notes.mid", DEFAULT)
    texts = [m.text for _, m in read_track(tmp_path / "notes.mid", "EVENTS")
             if m.type == "text"]
    assert "[end]" in texts


def test_variable_tempo_is_written_as_multiple_set_tempo(tmp_path):
    beats, step = [0.0], 0.5
    for _ in range(40):
        beats.append(beats[-1] + step)
        step *= 0.96
    ir = ChartIR(meta=SongMeta(name="t", duration=beats[-1]), tempo=TempoMap(beats),
                 instruments={"PART GUITAR": InstrumentChart(
                     "PART GUITAR", notes={"expert": [ChartNote(0, [0])]})})
    write_midi(ir, tmp_path / "notes.mid", DEFAULT)
    mid = mido.MidiFile(str(tmp_path / "notes.mid"))
    assert sum(1 for tr in mid.tracks for m in tr if m.type == "set_tempo") > 10


def test_rejects_ir_without_tempo_map(tmp_path):
    with pytest.raises(ValueError):
        write_midi(ChartIR(), tmp_path / "notes.mid", DEFAULT)

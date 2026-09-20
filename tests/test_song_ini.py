"""song.ini: tags, tipos e o que nao deve ser escrito."""

import configparser

from yargen.chart.ir import ChartIR, ChartNote, InstrumentChart, SongMeta, TempoMap
from yargen.config import DEFAULT
from yargen.meta.song_ini import build_tags, render, write_song_ini


def make_ir(n_notes=100, duration=120.0):
    tm = TempoMap.constant(120.0, duration)
    gtr = InstrumentChart("PART GUITAR", notes={
        "expert": [ChartNote(i * 240, [i % 5]) for i in range(n_notes)]})
    return ChartIR(meta=SongMeta(name="Musica", artist="Banda", duration=duration),
                   tempo=tm, instruments={"PART GUITAR": gtr})


def test_required_tags_are_present():
    tags = build_tags(make_ir(), DEFAULT)
    for key in ("name", "artist", "song_length", "diff_guitar",
                "preview_start_time", "delay", "icon", "charter"):
        assert key in tags


def test_song_length_is_in_milliseconds():
    assert build_tags(make_ir(duration=90.0), DEFAULT)["song_length"] == 90_000


def test_offset_goes_into_delay():
    cfg = DEFAULT.override_path("output.offset_ms", -40)
    assert build_tags(make_ir(), cfg)["delay"] == -40


def test_hopo_frequency_only_written_when_overridden():
    """Gravar o valor padrao mudaria o hash do chart sem mudar nada."""
    assert "hopo_frequency" not in build_tags(make_ir(), DEFAULT)
    cfg = DEFAULT.override_path("mapper.hopo_threshold_ticks", 170)
    assert build_tags(make_ir(), cfg)["hopo_frequency"] == 170


def test_difficulty_scales_with_density():
    sparse = build_tags(make_ir(20, 120.0), DEFAULT)["diff_guitar"]
    dense = build_tags(make_ir(900, 120.0), DEFAULT)["diff_guitar"]
    assert 0 <= sparse < dense <= 6


def test_diff_band_is_the_max_of_the_instruments():
    tags = build_tags(make_ir(900, 120.0), DEFAULT)
    assert tags["diff_band"] == tags["diff_guitar"]


def test_output_parses_as_ini(tmp_path):
    write_song_ini(make_ir(), tmp_path / "song.ini", DEFAULT)
    parser = configparser.ConfigParser()
    parser.read(tmp_path / "song.ini", encoding="utf-8-sig")
    assert parser["song"]["name"] == "Musica"
    assert parser["song"]["artist"] == "Banda"


def test_tag_order_is_stable():
    """Ordem estavel deixa o diff entre duas rodadas legivel."""
    ir = make_ir()
    assert render(build_tags(ir, DEFAULT)) == render(build_tags(ir, DEFAULT))
    assert render(build_tags(ir, DEFAULT)).splitlines()[0] == "[song]"
